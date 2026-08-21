"""
model_ctc.py
────────────────────────────────────────────────────────────────────────────
A separate model architecture from model.py's DigitFusionNet/AudioOnlyNet/
IMUOnlyNet — model.py's encoders deliberately collapse the whole time axis
into one embedding vector per trial (AdaptiveAvgPool to size 1) because
that pipeline always classifies one already-segmented letter/digit at a
time. A CTC model needs the opposite: keep (a downsampled version of) the
time axis all the way through, so the network can align an unsegmented,
multi-letter signal against a target letter sequence itself, instead of
requiring letter boundaries to be cut out in advance.

LetterCTCNet is trained on single-letter trials (target sequence length
always 1) and, at inference time, can be run on a whole, un-segmented word
span (first touch_on to last touch_off) — CTC's job is exactly to find
its own alignment between a long input sequence and a short target
sequence, which sidesteps needing per-letter touch boundaries inside a
word (unreliable — see dataset_ctc.py's docstring).

Architecture (see the chat this was built in for the reference diagram):

    Watch IMU segment  --[IMU encoder: TCN]--> z_imu  --[IMU decoder]--> fingertip trajectory
    Watch/surface mic  --[Mic encoder: U-Net]-> z_mic --[Mic decoder]--> reconstructed spectrogram
                                    \\              /
                                     [gated fusion]
                                          |
                                [sequence encoder: BiGRU]
                                          |
                                 per-frame letter probs
                                          |
                                     CTC decoding

Both encoders now have their OWN reconstruction decoder, reading directly
off that encoder's own latent (z_imu / z_mic) — BEFORE fusion, not after.
This is a deliberate change from an earlier version of this file, where a
single trajectory-decoding head sat AFTER the fused features (a "latent
fingertip motion" bottleneck the whole fused signal had to pass through).
Here each encoder is independently regularized to produce a
physically/acoustically meaningful latent — the IMU encoder's latent
should be enough to reconstruct real fingertip motion, the mic encoder's
latent should be enough to reconstruct the real spectrogram — with fusion
happening only afterward, on two latents that are each already anchored
to something real. See train_ctc.py's --motion-loss-weight and
--spec-loss-weight for the two auxiliary losses this trains against.

model.py is untouched; this file shares nothing with it at runtime beyond
both reading the same config.py constants (via config_ctc.py).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config_ctc


# ─── IMU encoder: Temporal Convolutional Network (TCN) ─────────────────────────

class _TCNBlock(nn.Module):
    """One dilated-residual TCN block: two same-padded dilated Conv1d
    layers (padding chosen so the time axis never shrinks) with a
    residual connection. Stacking blocks with GROWING dilation (1, 2, 4,
    ...) grows each timestep's effective receptive field exponentially
    without ever downsampling T — unlike the old IMUEncoderCTC (plain
    Conv1d + MaxPool1d, which shrank T by 4x), a TCN's whole point is
    "more context per timestep", not "fewer timesteps"."""

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2   # "same" padding — keeps T unchanged
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # (B, C, T) -> (B, C, T), same T
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.dropout(out)
        return self.relu(out + residual)


class IMUEncoderTCN(nn.Module):
    """Input: (B, C_in, T) variable-rate-resampled accel/gyro sequence ->
    z_imu: (B, T, hidden_channels). 3 dilated residual blocks (dilations
    1, 2, 4 by default, kernel size 3), matching this project's
    architecture diagram. Never downsamples T (TIME_DOWNSAMPLE=1) —
    every input timestep gets a corresponding output timestep, which is
    what lets the IMU decoder below reconstruct a trajectory at (close
    to) the same time resolution the real one was measured at."""
    TIME_DOWNSAMPLE = 1

    def __init__(self, in_channels: int, hidden_channels: int = 64, kernel_size: int = 3,
                 dilations: tuple = (1, 2, 4), dropout: float = 0.2):
        super().__init__()
        self.in_proj = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)   # lift to hidden_channels
        self.blocks = nn.ModuleList([
            _TCNBlock(hidden_channels, kernel_size, d, dropout) for d in dilations
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)          # (B, hidden, T)
        for block in self.blocks:
            x = block(x)             # (B, hidden, T), T unchanged throughout
        return x.transpose(1, 2)     # (B, T, hidden)


class IMUDecoder(nn.Module):
    """z_imu -> reconstructed (x, y) fingertip trajectory at every
    timestep — reads DIRECTLY off the IMU encoder's own latent, before
    any fusion with audio. Trained with an auxiliary MSE loss against the
    real, NORMALIZED trajectory (see dataset_ctc.load_trajectory_variable
    — normalized meaning no absolute world coordinates, only the
    relative shape of the motion). A single Linear layer is enough here:
    the point isn't to give this decoder much capacity of its own, it's
    to force z_imu ITSELF to already contain what's needed to reconstruct
    real motion, which is exactly what a weak decoder + a real target
    achieves — a strong decoder could compensate for a z_imu that
    doesn't actually encode motion well, defeating the purpose."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.proj = nn.Linear(in_channels, 2)

    def forward(self, z_imu: torch.Tensor) -> torch.Tensor:   # (B, T, C) -> (B, T, 2)
        return self.proj(z_imu)


# ─── Mic encoder: U-Net ─────────────────────────────────────────────────────────

class UNetMicEncoder(nn.Module):
    """Input: (B, 1, N_MELS, T) log-mel spectrogram -> (z_mic, recon).
    z_mic: (B, T', out_channels) — the bottleneck (deepest encoder
    layer), frequency-pooled and fed to fusion, same shape convention as
    the old AudioEncoderCTC. recon: (B, 1, N_MELS, T) — the full
    encoder-decoder path's reconstruction of the INPUT spectrogram
    itself, via skip connections from each encoder stage to its matching
    decoder stage (standard U-Net) — this is the "Mic decoder" auxiliary
    task (see train_ctc.py's --spec-loss-weight): reconstructing the
    input FORCES the encoder's intermediate features to preserve
    information a naive classification-only objective might discard.

    Scaled down from the architecture diagram's 5 encoder / 5 decoder
    stages to 3 encoder / 2 pooling stages: this project's spectrograms
    are much shorter (often under 100 frames for a single letter) and
    narrower (64 mel bins) than the full speech utterances a 5-stage
    U-Net is normally sized for — 5 stages of stride-2 pooling would need
    at least 32 in each dimension just to avoid collapsing to zero elsewhere
    in the pipeline, which isn't safely true here. Skip-connected upsampling
    targets each stage's EXACT spatial size via interpolation (not a fixed
    stride-2 transpose conv) specifically so odd/non-power-of-2 time lengths
    (routine here, since duration is never fixed) never cause a shape
    mismatch at the concatenation step."""
    TIME_DOWNSAMPLE = 4   # two stride-2 poolings in the encoder path below

    def __init__(self, out_channels: int = 64, base_channels: int = 16):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(1, base_channels, 3, padding=1),
                                   nn.BatchNorm2d(base_channels), nn.ReLU())
        self.pool1 = nn.MaxPool2d((2, 2))
        self.enc2 = nn.Sequential(nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
                                   nn.BatchNorm2d(base_channels * 2), nn.ReLU())
        self.pool2 = nn.MaxPool2d((2, 2))
        self.enc3 = nn.Sequential(nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1),
                                   nn.BatchNorm2d(base_channels * 4), nn.ReLU())

        self.to_latent = nn.Conv2d(base_channels * 4, out_channels, kernel_size=1)
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))   # collapse remaining freq to 1 for z_mic

        # Decoder path — each stage upsamples to its matching skip
        # tensor's EXACT size (via interpolate, in forward()) before a
        # conv processes the concatenation, so this never depends on
        # T being a multiple of 4.
        self.dec2_conv = nn.Sequential(
            nn.Conv2d(base_channels * 4 + base_channels * 2, base_channels * 2, 3, padding=1),
            nn.BatchNorm2d(base_channels * 2), nn.ReLU())
        self.dec1_conv = nn.Sequential(
            nn.Conv2d(base_channels * 2 + base_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels), nn.ReLU())
        self.recon_head = nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor):
        e1 = self.enc1(x)              # (B, base, F, T)
        p1 = self.pool1(e1)            # (B, base, F/2, T/2) roughly
        e2 = self.enc2(p1)             # (B, base*2, F/2, T/2)
        p2 = self.pool2(e2)            # (B, base*2, F/4, T/4) roughly
        e3 = self.enc3(p2)             # (B, base*4, F/4, T/4)

        latent = self.to_latent(e3)             # (B, out_channels, F/4, T/4)
        z = self.freq_pool(latent)              # (B, out_channels, 1, T/4)
        z = z.squeeze(2).transpose(1, 2)        # (B, T/4, out_channels)

        u2 = F.interpolate(e3, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec2_conv(torch.cat([u2, e2], dim=1))     # (B, base*2, F/2, T/2)
        u1 = F.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.dec1_conv(torch.cat([u1, e1], dim=1))     # (B, base, F, T)
        recon = self.recon_head(d1)                          # (B, 1, F, T) — matches input's exact shape

        return z, recon


def _downsampled_length(lengths: torch.Tensor, factor: int, actual_T: int) -> torch.Tensor:
    """Approximates each sample's true valid-timestep count after the
    encoder's pooling, from its ORIGINAL (pre-padding) length — needed
    because CTC must never be told a padded trial is longer than it
    really is (that would let the loss/decoder read blank-padded noise
    as if it were real signal). Clamped to actual_T as a safety net since
    MaxPool1d's exact output length has some boundary-condition rounding
    this only approximates."""
    out = torch.div(lengths, factor, rounding_mode='floor').clamp(min=1)
    return torch.clamp(out, max=actual_T)


class LetterCTCNet(nn.Module):
    """modality: 'audio' / 'imu' / 'fusion' — same three options as
    model.py's MODALITIES, same meaning. n_classes is the letter alphabet
    size (26) — the classifier head is n_classes+1, index 0 reserved for
    CTC's blank symbol (see config_ctc.BLANK_IDX)."""

    def __init__(self, modality: str, n_classes: int, imu_in_channels: int,
                 audio_embed: int = 64, imu_embed: int = 64, rnn_hidden: int = 256,
                 dropout: float = 0.3, use_space: bool = True):
        super().__init__()
        if modality not in ('audio', 'imu', 'fusion'):
            raise ValueError(f'modality must be audio/imu/fusion, got "{modality}"')
        self.modality = modality
        self.use_space = use_space

        if modality in ('audio', 'fusion'):
            self.audio_enc = UNetMicEncoder(out_channels=audio_embed)
        if modality in ('imu', 'fusion'):
            self.imu_enc = IMUEncoderTCN(imu_in_channels, hidden_channels=imu_embed)
            self.imu_decoder = IMUDecoder(imu_embed)   # reads z_imu directly, BEFORE fusion

        if modality == 'fusion':
            # Gated fusion: audio/IMU latents are each projected to a
            # shared fusion_dim, then a per-timestep, per-channel gate
            # (learned from BOTH streams together) decides how much of
            # each to keep at that instant — feat = gate*audio +
            # (1-gate)*imu — instead of just handing the RNN both
            # streams side by side and hoping it learns to weigh them
            # implicitly. fusion_dim = max(audio_embed, imu_embed) so
            # this still works if the two embed sizes are ever set
            # differently (not just the equal defaults below).
            fusion_dim = max(audio_embed, imu_embed)
            self.audio_gate_proj = nn.Linear(audio_embed, fusion_dim)
            self.imu_gate_proj = nn.Linear(imu_embed, fusion_dim)
            self.gate_fc = nn.Linear(fusion_dim * 2, fusion_dim)
            rnn_in = fusion_dim
        elif modality == 'audio':
            rnn_in = audio_embed
        else:
            rnn_in = imu_embed

        self.rnn = nn.GRU(rnn_in, rnn_hidden, num_layers=2, batch_first=True,
                           bidirectional=True, dropout=dropout)
        # Dropout right before the classifier head too, not just inside the
        # GRU stack — added specifically because a small/imbalanced letter
        # dataset (as few as 3-5 trials for some letters) gives this model
        # (hundreds of thousands of params) plenty of room to just memorize
        # the training set rather than generalize; see the chat this was
        # added in for the train/val loss curves that showed exactly that.
        self.dropout = nn.Dropout(dropout)
        # +2 (not +1) only when use_space is True: index 0 is CTC's blank
        # (config_ctc.BLANK_IDX), indices 1..n_classes are the letters,
        # and — only if use_space — index n_classes+1 is a dedicated
        # SPACE class, so sentence targets can keep their real word
        # boundaries instead of having spaces silently stripped out
        # before training. See ctc_greedy_decode()/ctc_beam_search_decode()
        # below for how this index gets turned back into a literal ' ' at
        # decode time.
        #
        # use_space=False is a deliberate, testable ALTERNATIVE, not a
        # legacy fallback: space is fundamentally a different KIND of
        # signal than a letter — it's an absence of writing motion/sound
        # rather than a positive acoustic/motion signature the way each
        # letter has — and forcing the same classifier to discriminate
        # "no letter right now" (blank) from "a pause between words"
        # (space) from 26 real letters may make letter discrimination
        # itself harder, not easier, since blank and space overlap in
        # meaning far more than any two letters do. With use_space=False,
        # sentence targets fall back to a continuous run of letters
        # (spaces stripped, same as before space support existed) — see
        # dataset_ctc_realtext.py's own use_space flag, which must be set
        # the same way for a given checkpoint's training data to match
        # what this model was actually built to output.
        self.classifier = nn.Linear(rnn_hidden * 2, n_classes + 2 if use_space else n_classes + 1)

        self.last_gate = None   # (B, T', fusion_dim) from the most recent forward() call, fusion
                                 # modality only — purely diagnostic (e.g. to later visualize which
                                 # stream the model leaned on when), never read by anything else here

    def forward(self, audio: "torch.Tensor | None", audio_lengths: "torch.Tensor | None",
                imu: "torch.Tensor | None", imu_lengths: "torch.Tensor | None"):
        """Returns (log_probs, output_lengths, traj_pred, spec_recon):
          - log_probs: (T', B, n_classes+1), CTC's expected time-major layout
          - output_lengths: (B,), each sample's real (unpadded) timestep
            count after the encoder's downsampling
          - traj_pred: (B, T_imu, 2) or None (modality='audio') — the IMU
            decoder's reconstructed (x, y) trajectory, read directly off
            z_imu at IMU's OWN native time resolution (not downsampled to
            match audio/fusion — see IMUEncoderTCN's TIME_DOWNSAMPLE=1)
          - spec_recon: (B, 1, N_MELS, T_audio) or None (modality='imu')
            — the Mic decoder's reconstructed spectrogram, at audio's own
            original (pre-encoder) shape
        Callers that don't use either auxiliary task (--motion-loss-weight=0
        / --spec-loss-weight=0) can simply ignore these two return values."""
        traj_pred = None
        spec_recon = None

        if self.modality == 'audio':
            feat, spec_recon = self.audio_enc(audio)
            out_lengths = _downsampled_length(audio_lengths, UNetMicEncoder.TIME_DOWNSAMPLE, feat.shape[1])
        elif self.modality == 'imu':
            feat = self.imu_enc(imu)                    # (B, T_imu, C) — T_imu unchanged from input
            traj_pred = self.imu_decoder(feat)
            out_lengths = _downsampled_length(imu_lengths, IMUEncoderTCN.TIME_DOWNSAMPLE, feat.shape[1])
        else:
            a_feat, spec_recon = self.audio_enc(audio)   # (B, T_a', C_a)
            i_feat = self.imu_enc(imu)                    # (B, T_imu, C_i) — T_imu unchanged
            traj_pred = self.imu_decoder(i_feat)           # decoded at IMU's OWN time resolution,
                                                              # BEFORE the interpolation below
            # Audio and IMU have different native sample rates, so their
            # encoders produce different T' even for the same trial —
            # interpolating IMU's time axis onto audio's (an arbitrary but
            # consistent choice of "reference clock") lets the two be
            # combined per-timestep before the shared RNN. This
            # interpolated COPY is only used for fusion — traj_pred above
            # was already computed from the un-interpolated i_feat, so
            # the trajectory reconstruction isn't affected by this
            # resampling.
            i_feat_aligned = F.interpolate(
                i_feat.transpose(1, 2), size=a_feat.shape[1], mode='linear', align_corners=False,
            ).transpose(1, 2)

            a_proj = self.audio_gate_proj(a_feat)            # (B, T', fusion_dim)
            i_proj = self.imu_gate_proj(i_feat_aligned)       # (B, T', fusion_dim)
            gate = torch.sigmoid(self.gate_fc(torch.cat([a_proj, i_proj], dim=2)))   # (B, T', fusion_dim)
            feat = gate * a_proj + (1 - gate) * i_proj        # per-timestep, per-channel blend
            self.last_gate = gate.detach()
            out_lengths = _downsampled_length(audio_lengths, UNetMicEncoder.TIME_DOWNSAMPLE, feat.shape[1])

        rnn_out, _ = self.rnn(feat)               # (B, T', 2*rnn_hidden)
        rnn_out = self.dropout(rnn_out)
        logits = self.classifier(rnn_out)          # (B, T', n_classes+1)
        log_probs = F.log_softmax(logits, dim=2)
        return log_probs.transpose(0, 1), out_lengths, traj_pred, spec_recon


def build_model_ctc(modality: str, n_classes: int,
                     imu_in_channels: int = len(config_ctc.IMU_CHANNELS),
                     dropout: float = 0.3, rnn_hidden: int = 256, use_space: bool = True) -> LetterCTCNet:
    return LetterCTCNet(modality, n_classes, imu_in_channels, dropout=dropout, rnn_hidden=rnn_hidden,
                         use_space=use_space)


def ctc_greedy_decode(log_probs: torch.Tensor, lengths: torch.Tensor, classes: list,
                       use_space: bool = True) -> list:
    """Standard CTC greedy decode: argmax every timestep, collapse
    consecutive repeats, drop blanks. log_probs: (T, B, n_classes+2-or-+1)
    as returned by LetterCTCNet.forward() — the +1 vs +2 depends on
    whether the model this came from was built with use_space (must
    match — see LetterCTCNet's classifier comment for why). When
    use_space, index len(classes)+1 is the dedicated SPACE class and
    decodes to a literal ' '; when not, that index doesn't exist at all
    (the classifier only has n_classes+1 outputs) and this parameter
    just skips ever looking for it. Returns a list of B decoded strings;
    classes[i] is the letter for label index i+1."""
    space_idx = len(classes) + 1 if use_space else None
    preds = log_probs.argmax(dim=2).transpose(0, 1)   # (B, T)
    results = []
    for b in range(preds.shape[0]):
        seq = preds[b, :int(lengths[b])].tolist()
        collapsed = []
        prev = None
        for idx in seq:
            if idx != prev:
                if idx == config_ctc.BLANK_IDX:
                    pass
                elif space_idx is not None and idx == space_idx:
                    collapsed.append(' ')
                else:
                    collapsed.append(classes[idx - 1])
                prev = idx
        results.append(''.join(collapsed))
    return results


def ctc_beam_search_decode(log_probs: torch.Tensor, lengths: torch.Tensor, classes: list,
                            beam_width: int = 10, use_space: bool = True) -> list:
    """CTC prefix beam search decode — unlike ctc_greedy_decode (which
    commits to the single highest-probability symbol at every timestep
    and can never recover from an early mistake), this keeps the
    beam_width most probable PREFIXES (partial decoded strings) alive at
    every timestep, extends each one under every possible next symbol,
    and re-prunes back down to beam_width after each step. The final
    answer is whichever surviving prefix has the highest total
    probability once the whole sequence has been processed. This matters
    most for longer sequences (sentences especially) — a single wrong
    argmax early in a long trial derails greedy decode for everything
    after it, while beam search keeps enough alternative hypotheses
    alive to recover.

    Standard CTC prefix beam search (Hannun et al.): each surviving
    prefix tracks TWO separate probabilities — p_blank (probability mass
    of paths ending in this prefix where the last emitted symbol was
    blank) and p_non_blank (same, but the last emitted symbol was a real
    character) — kept apart because whether a REPEATED character extends
    the prefix (new letter) or just re-affirms it (same letter, still
    "open") depends on whether a blank came between the two occurrences,
    which is exactly what p_blank vs p_non_blank distinguishes.

    log_probs: (T, B, n_classes+2-or-+1) as returned by LetterCTCNet.forward()
    — same +2-vs-+1 convention as ctc_greedy_decode (index 0 = blank,
    index len(classes)+1 = space, only when use_space matches how the
    model was built). Returns a list of B decoded strings — the single
    best (highest-probability) hypothesis per sample. beam_width trades
    decode time for search thoroughness — larger finds better hypotheses
    but takes proportionally longer; beam_width=1 degenerates to (a
    slightly slower, but equivalent) greedy decode."""
    space_idx = len(classes) + 1 if use_space else None
    B = log_probs.shape[1]
    probs_all = log_probs.exp().cpu().numpy()   # (T, B, n_out) — plain probabilities; T is small
                                                   # enough in this project (a few hundred at most)
                                                   # that log-space bookkeeping isn't needed here
    results = []
    for b in range(B):
        T = int(lengths[b])
        probs = probs_all[:T, b, :]   # (T, n_out)
        # beams: {prefix (tuple of class indices) -> (p_blank, p_non_blank)}
        beams = {(): (1.0, 0.0)}
        for t in range(T):
            frame = probs[t]
            next_beams = {}
            for prefix, (p_b, p_nb) in beams.items():
                p_total = p_b + p_nb
                last = prefix[-1] if prefix else None
                for c in range(frame.shape[0]):
                    p_c = frame[c]
                    if p_c <= 0.0:
                        continue
                    if c == config_ctc.BLANK_IDX:
                        nb_b, nb_nb = next_beams.get(prefix, (0.0, 0.0))
                        next_beams[prefix] = (nb_b + p_total * p_c, nb_nb)
                    elif c == last:
                        # same symbol as the prefix's last character: without a blank in
                        # between, this just re-affirms the currently-open symbol (no new
                        # character) — extends via p_nb only, same prefix
                        nb_b, nb_nb = next_beams.get(prefix, (0.0, 0.0))
                        next_beams[prefix] = (nb_b, nb_nb + p_nb * p_c)
                        # but arriving at this symbol AFTER a blank (p_b branch) IS a
                        # genuinely new occurrence of the same letter (e.g. the two o's
                        # in "book") — that extends to a longer, new prefix
                        new_prefix = prefix + (c,)
                        nb_b2, nb_nb2 = next_beams.get(new_prefix, (0.0, 0.0))
                        next_beams[new_prefix] = (nb_b2, nb_nb2 + p_b * p_c)
                    else:
                        new_prefix = prefix + (c,)
                        nb_b, nb_nb = next_beams.get(new_prefix, (0.0, 0.0))
                        next_beams[new_prefix] = (nb_b, nb_nb + p_total * p_c)
            # prune to the beam_width highest-total-probability prefixes
            beams = dict(sorted(next_beams.items(), key=lambda kv: -(kv[1][0] + kv[1][1]))[:beam_width])
            if not beams:
                beams = {(): (1.0, 0.0)}   # defensive fallback — shouldn't normally happen since
                                             # every frame's probabilities sum to ~1 across classes
        best_prefix = max(beams.items(), key=lambda kv: kv[1][0] + kv[1][1])[0]
        chars = []
        for c in best_prefix:
            if space_idx is not None and c == space_idx:
                chars.append(' ')
            else:
                chars.append(classes[c - 1])
        results.append(''.join(chars))
    return results
