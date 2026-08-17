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

model.py is untouched; this file shares nothing with it at runtime beyond
both reading the same config.py constants (via config_ctc.py).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config_ctc


class AudioEncoderCTC(nn.Module):
    """Input: (B, 1, N_MELS, T) log-mel spectrogram -> (B, T', C), keeping
    a downsampled time axis instead of collapsing it. Frequency is pooled
    away completely (same intent as model.py's AudioEncoder, just only
    over frequency). TIME_DOWNSAMPLE is the single source of truth for how
    much shorter T' is than T — LetterCTCNet.forward() uses it to keep
    output_lengths consistent with the actual tensor shape."""
    TIME_DOWNSAMPLE = 4   # two stride-2 poolings below

    def __init__(self, out_channels: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(),
        )
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))   # collapse whatever freq dim remains to
                                                            # 1, leave the time dim exactly as-is

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)          # (B, C, F', T')
        x = self.freq_pool(x)     # (B, C, 1, T')
        x = x.squeeze(2)          # (B, C, T')
        return x.transpose(1, 2)  # (B, T', C)


class IMUEncoderCTC(nn.Module):
    """Input: (B, C_in, T) variable-rate-resampled accel/gyro sequence ->
    (B, T', C_out), same "keep time, downsample it a known amount" idea as
    AudioEncoderCTC."""
    TIME_DOWNSAMPLE = 4   # two stride-2 poolings below

    def __init__(self, in_channels: int, out_channels: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2), nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, out_channels, kernel_size=5, padding=2), nn.BatchNorm1d(out_channels), nn.ReLU(),
            nn.MaxPool1d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)          # (B, C, T')
        return x.transpose(1, 2)  # (B, T', C)


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
                 audio_embed: int = 64, imu_embed: int = 64, rnn_hidden: int = 128,
                 dropout: float = 0.3):
        super().__init__()
        if modality not in ('audio', 'imu', 'fusion'):
            raise ValueError(f'modality must be audio/imu/fusion, got "{modality}"')
        self.modality = modality

        if modality in ('audio', 'fusion'):
            self.audio_enc = AudioEncoderCTC(out_channels=audio_embed)
        if modality in ('imu', 'fusion'):
            self.imu_enc = IMUEncoderCTC(imu_in_channels, out_channels=imu_embed)

        if modality == 'fusion':
            # Gated fusion, replacing plain concatenation: audio/IMU
            # features are each projected to a shared fusion_dim, then a
            # per-timestep, per-channel gate (learned from BOTH streams
            # together) decides how much of each to keep at that instant
            # — feat = gate*audio + (1-gate)*imu — instead of just handing
            # the RNN both streams side by side and hoping it learns to
            # weigh them implicitly. Motivation: a fixed concatenation
            # gives the audio and IMU streams equal say everywhere, even
            # at moments where one is much more informative than the
            # other (e.g. a stretch where the fingertip camera briefly
            # loses tracking) — a gate lets the network down-weight
            # whichever stream is less reliable AT THAT MOMENT, rather
            # than only ever learning one fixed blend for the whole
            # trial. fusion_dim = max(audio_embed, imu_embed) so this
            # still works if the two embed sizes are ever set differently
            # (not just the equal defaults below).
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
        self.classifier = nn.Linear(rnn_hidden * 2, n_classes + 1)

        self.last_gate = None   # (B, T', fusion_dim) from the most recent forward() call, fusion
                                 # modality only — purely diagnostic (e.g. to later visualize which
                                 # stream the model leaned on when), never read by anything else here

    def forward(self, audio: "torch.Tensor | None", audio_lengths: "torch.Tensor | None",
                imu: "torch.Tensor | None", imu_lengths: "torch.Tensor | None"):
        """Returns (log_probs, output_lengths): log_probs is (T', B,
        n_classes+1) — CTC's expected (time-major) layout — and
        output_lengths is (B,), each sample's real (unpadded) timestep
        count after the encoder's downsampling."""
        if self.modality == 'audio':
            feat = self.audio_enc(audio)
            out_lengths = _downsampled_length(audio_lengths, AudioEncoderCTC.TIME_DOWNSAMPLE, feat.shape[1])
        elif self.modality == 'imu':
            feat = self.imu_enc(imu)
            out_lengths = _downsampled_length(imu_lengths, IMUEncoderCTC.TIME_DOWNSAMPLE, feat.shape[1])
        else:
            a_feat = self.audio_enc(audio)   # (B, T_a', C_a)
            i_feat = self.imu_enc(imu)       # (B, T_i', C_i)
            # Audio and IMU have different native sample rates, so their
            # encoders produce different T' even for the same trial —
            # interpolating IMU's time axis onto audio's (an arbitrary but
            # consistent choice of "reference clock") lets the two be
            # combined per-timestep before the shared RNN.
            i_feat_aligned = F.interpolate(
                i_feat.transpose(1, 2), size=a_feat.shape[1], mode='linear', align_corners=False,
            ).transpose(1, 2)

            a_proj = self.audio_gate_proj(a_feat)            # (B, T', fusion_dim)
            i_proj = self.imu_gate_proj(i_feat_aligned)       # (B, T', fusion_dim)
            gate = torch.sigmoid(self.gate_fc(torch.cat([a_proj, i_proj], dim=2)))   # (B, T', fusion_dim)
            feat = gate * a_proj + (1 - gate) * i_proj        # per-timestep, per-channel blend
            self.last_gate = gate.detach()
            out_lengths = _downsampled_length(audio_lengths, AudioEncoderCTC.TIME_DOWNSAMPLE, feat.shape[1])

        rnn_out, _ = self.rnn(feat)               # (B, T', 2*rnn_hidden)
        rnn_out = self.dropout(rnn_out)
        logits = self.classifier(rnn_out)          # (B, T', n_classes+1)
        log_probs = F.log_softmax(logits, dim=2)
        return log_probs.transpose(0, 1), out_lengths   # (T', B, n_classes+1), (B,)


def build_model_ctc(modality: str, n_classes: int,
                     imu_in_channels: int = len(config_ctc.IMU_CHANNELS),
                     dropout: float = 0.3) -> LetterCTCNet:
    return LetterCTCNet(modality, n_classes, imu_in_channels, dropout=dropout)


def ctc_greedy_decode(log_probs: torch.Tensor, lengths: torch.Tensor, classes: list) -> list:
    """Standard CTC greedy decode: argmax every timestep, collapse
    consecutive repeats, drop blanks. log_probs: (T, B, n_classes+1) as
    returned by LetterCTCNet.forward(). Returns a list of B decoded
    strings. classes[i] is the letter for label index i+1 (index 0 is
    always blank — see config_ctc.BLANK_IDX)."""
    preds = log_probs.argmax(dim=2).transpose(0, 1)   # (B, T)
    results = []
    for b in range(preds.shape[0]):
        seq = preds[b, :int(lengths[b])].tolist()
        collapsed = []
        prev = None
        for idx in seq:
            if idx != prev:
                if idx != config_ctc.BLANK_IDX:
                    collapsed.append(classes[idx - 1])
                prev = idx
        results.append(''.join(collapsed))
    return results