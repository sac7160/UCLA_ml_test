"""
model_writeas.py
────────────────────────────────────────────────────────────────────────────
Baseline replication of WriteAS (Zhang et al., IMWUT 2021 — "Write,
Attend and Spell: Streaming End-to-end Free-style Handwriting Recognition
Using Smartwatches") for direct comparison against this project's own
architecture, on the SAME data.

Three components replicated, matching the paper's own Section 4:
1. Multimodal CNN (Section 4.2 / Fig. 5): learns MODALITY-SPECIFIC
   features per short overlapping clip first, THEN fuses across
   modalities via merged conv layers — rather than concatenating raw
   sensor channels upfront the way a single shared encoder would.
2. Stacked dilated CAUSAL convolution encoder (Section 4.3.1): WaveNet-
   style [van den Oord et al. 2016] gated activation units, 6 layers,
   dilation schedule exactly as stated in the paper's own text — 1, 2,
   5, 2, 2, 5 — kernel_size=5, 128 filters/layer. Causal (never looks at
   future timesteps) — this is what makes the paper's STREAMING claim
   possible; this project's own TCN encoder (model_ctc.py) is NOT causal.
3. Multi-task CTC + attention decoder (Section 4.3.2-4.3.3): the CTC
   head gives streaming per-clip character probabilities directly off
   the shared encoder; a separate GRU + Bahdanau-attention decoder
   re-scores the whole word once it's finished. Combined training loss:
   L_word = (1-lambda)*L_attention + lambda*L_CTC, lambda=0.8 (the
   paper's own tuned value, Section 4.3.3).

ADAPTATIONS made for this project's data (flagged here explicitly, not
silently):
- The paper's 3 sensor modalities are linear accelerometer (gravity
  REMOVED from raw accel), gyroscope, and a SEPARATE gravity sensor —
  three genuinely different signals (Section 3.1). This project's own
  IMU data (accel_x/y/z, gyro_x/y/z — see config_ctc.IMU_CHANNELS) has
  no separate gravity channel recorded, so MultimodalCNN here uses 2
  branches (accel, gyro) instead of 3. If a genuine gravity channel is
  ever added to this project's own data, adding a third branch is a
  small, localized change (see MultimodalCNN.__init__'s branch count).
- The paper's system is IMU-only (no audio) — this replication is
  IMU-only too, for a fair like-for-like comparison against this
  project's own --modality imu runs, not its --modality fusion runs.
- Clip windowing: the paper segments each word-level trial into
  overlapping T=0.3s clips (50% overlap) BEFORE the multimodal CNN —
  this file implements that explicitly (see _window_clips), unlike this
  project's own TCN encoder, which streams the whole variable-length
  IMU signal through directly with no clip segmentation step.
- <EOS> early-stopping during autoregressive INFERENCE (not training) is
  not implemented — generation always runs to --max-decode-len and the
  caller trims at the first predicted <EOS> token afterward (see
  evaluate_writeas.py). This keeps the decoding loop simple without
  changing what the model itself learns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config_ctc


def _window_clips(x: torch.Tensor, lengths: torch.Tensor, clip_sec: float, overlap: float,
                   sample_rate: float):
    """(B, C, T) -> (clips (B, n_clips, C, clip_len), n_clips_per_sample (B,)) via
    overlapping sliding windows — see this file's own docstring / WriteAS
    Section 4.2: "the word-level signal stream is first segmented into a
    sequence of overlapping clips with length T, where T is 0.3s and
    overlapping is 50%". Built via unfold() after right-padding with
    zeros so every trial (even one shorter than a single clip) still
    produces at least one clip. n_clips_per_sample tells the caller how
    many of each sample's clips are real (not built entirely from
    padding), for masking."""
    clip_len = max(1, int(round(clip_sec * sample_rate)))
    hop = max(1, int(round(clip_len * (1 - overlap))))
    B, C, T = x.shape
    if T < clip_len:
        x = F.pad(x, (0, clip_len - T))
        T = clip_len
    n_clips = max(1, (T - clip_len) // hop + 1)
    pad_needed = (n_clips - 1) * hop + clip_len - T
    if pad_needed > 0:
        x = F.pad(x, (0, pad_needed))
    clips = x.unfold(dimension=2, size=clip_len, step=hop)   # (B, C, n_clips, clip_len)
    clips = clips.permute(0, 2, 1, 3).contiguous()             # (B, n_clips, C, clip_len)
    n_clips_per_sample = torch.clamp(
        ((lengths.float() - clip_len) / hop).floor().long() + 1, min=1, max=n_clips)
    return clips, n_clips_per_sample


class _ModalityBranch(nn.Module):
    """One "Individual Conv" branch from Fig. 5 — 3-layer 2D CNN over one
    modality's (clip_len, 3-axis) clip. Kernel/pool/filter sizes exactly
    as given in the figure: (5x3, 2x1, 8) -> (5x3, 2x1, 16) -> (5x3, 1x1,
    16) — the first two layers pool along time (2x1) to reduce
    dimensionality, the third does not (1x1 = no-op pool, kept only for
    symmetry with the paper's own notation in Fig. 5)."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(5, 3), padding=(2, 1)),
            nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d((2, 1)))
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=(5, 3), padding=(2, 1)),
            nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d((2, 1)))
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=(5, 3), padding=(2, 1)),
            nn.BatchNorm2d(16), nn.ReLU())   # no pooling — Fig. 5's own "1x1" pool size

    def forward(self, x):   # x: (B, 1, clip_len, 3) — "3" is this modality's own axis count
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x   # (B, 16, clip_len/4, 3)


class MultimodalCNN(nn.Module):
    """Section 4.2 / Fig. 5 — one _ModalityBranch per sensor modality
    (accel, gyro here — see this file's own top-level docstring for why
    2, not the paper's 3), NO weight sharing across branches, outputs
    concatenated along the channel dimension, then 2 "Merged Conv"
    layers (3x3, 1x1, 48 filters — Fig. 5's own values) learn cross-
    modality features, flattened and projected to a 32-dim vector per
    clip (Fig. 5's own "vector length = 32")."""

    def __init__(self, n_modalities: int = 2, out_dim: int = 32):
        super().__init__()
        self.branches = nn.ModuleList([_ModalityBranch() for _ in range(n_modalities)])
        merged_in = 16 * n_modalities
        self.merged_conv1 = nn.Sequential(
            nn.Conv2d(merged_in, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48), nn.ReLU())
        self.merged_conv2 = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48), nn.ReLU())
        self.dropout = nn.Dropout(0.4)   # paper's own stated value (Section 4.2), applied after
                                           # the FC projection below
        self.fc = nn.LazyLinear(out_dim)   # flattened size depends on clip_len — resolved on
                                             # the first real forward() call

    def forward(self, modality_clips: list) -> torch.Tensor:   # each entry: (B, 1, clip_len, axes)
        feats = [branch(clip) for branch, clip in zip(self.branches, modality_clips)]
        merged = torch.cat(feats, dim=1)                # (B, 16*n_modalities, clip_len/4, axes)
        merged = self.merged_conv1(merged)
        merged = self.merged_conv2(merged)                # (B, 48, clip_len/4, axes)
        flat = merged.flatten(start_dim=1)
        out = self.dropout(self.fc(flat))
        return out   # (B, out_dim)


class DilatedCausalConvEncoder(nn.Module):
    """Section 4.3.1 — WaveNet-style [van den Oord et al. 2016] stacked
    dilated CAUSAL convolutions. 6 layers, dilation schedule exactly as
    the paper's own text states: 1, 2, 5, 2, 2, 5. kernel_size=5, 128
    filters/layer (paper's own stated values).

    Per layer: z = tanh(dilated_conv(h)) * sigmoid(dilated_conv(h))
    (gated activation unit, Eq. 5); o_i = tanh(1x1_conv(z)) (Eq. 6);
    h = h + o_i (residual, Eq. 7). The encoder's actual OUTPUT is the SUM
    of every layer's o_i (Eq. 8 — "the outputs on of the dilated
    convolutional network are the sums of o^i_n for all dilated layers"),
    not just the last layer's — both the CTC head and attention decoder
    read this summed value.

    Causal: each dilated conv is padded on the LEFT ONLY by
    (kernel_size-1)*dilation, then the output is trimmed back to the
    input's own length — this guarantees position t's output never
    depends on any input at a position > t, unlike a standard
    (centered-padding) causal-agnostic convolution."""
    DILATIONS = (1, 2, 5, 2, 2, 5)

    def __init__(self, channels: int = 128, kernel_size: int = 5):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilated_convs = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size, dilation=d, padding=0)
            for d in self.DILATIONS
        ])
        self.skip_convs = nn.ModuleList([nn.Conv1d(channels, channels, 1) for _ in self.DILATIONS])

    def forward(self, h: torch.Tensor) -> torch.Tensor:   # h: (B, channels, T)
        skip_sum = 0.0
        for dilated_conv, skip_conv, d in zip(self.dilated_convs, self.skip_convs, self.DILATIONS):
            left_pad = (self.kernel_size - 1) * d
            h_padded = F.pad(h, (left_pad, 0))          # LEFT-only padding -> causal
            conv_out = dilated_conv(h_padded)              # (B, channels, T) — exactly T again,
                                                              # thanks to the left-pad amount
            z = torch.tanh(conv_out) * torch.sigmoid(conv_out)   # Eq. 5
            o_i = torch.tanh(skip_conv(z))                        # Eq. 6
            h = h + o_i                                            # Eq. 7 (residual)
            skip_sum = skip_sum + o_i                              # Eq. 8 (summed skip connections)
        return skip_sum   # (B, channels, T) — the encoder's actual output per Eq. 8


class BahdanauAttention(nn.Module):
    """Standard additive attention [Bahdanau et al. 2015 — cited as [14]
    in the paper] — scores every encoder timestep against the decoder's
    current hidden state, softmax-normalizes into weights, returns the
    weighted-sum context vector (Section 4.3.2's own alpha^n_u, c_u)."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.energy_proj = nn.Linear(hidden_dim, 1)

    def forward(self, decoder_hidden: torch.Tensor, encoder_outputs: torch.Tensor,
                encoder_mask: torch.Tensor):
        # decoder_hidden: (B, H); encoder_outputs: (B, T, H); encoder_mask: (B, T) bool
        query = self.query_proj(decoder_hidden).unsqueeze(1)              # (B, 1, H)
        keys = self.key_proj(encoder_outputs)                              # (B, T, H)
        energy = self.energy_proj(torch.tanh(query + keys)).squeeze(-1)    # (B, T)
        energy = energy.masked_fill(~encoder_mask, float('-inf'))
        alpha = F.softmax(energy, dim=1)                                    # (B, T)
        context = torch.bmm(alpha.unsqueeze(1), encoder_outputs).squeeze(1)   # (B, H)
        return context, alpha


class AttentionDecoder(nn.Module):
    """Section 4.3.2 — 2-layer GRU, 128 cells/layer (paper's own stated
    values), letter embedding (one-hot -> 12-dim dense, paper's own
    stated value), Bahdanau attention over the FULL encoder output
    sequence at every decode step. Autoregressive: at each step reads the
    previous letter's embedding + the current attention context vector,
    predicts the next letter — matching Eq. 9's own formulation. <BOS>/
    <EOS> are extra classes appended after the real alphabet (see
    WriteASModel's own class-index convention)."""

    def __init__(self, n_classes_with_specials: int, hidden_dim: int = 128,
                 embed_dim: int = 12, n_layers: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(n_classes_with_specials, embed_dim)
        self.attention = BahdanauAttention(hidden_dim)
        self.gru = nn.GRU(embed_dim + hidden_dim, hidden_dim, num_layers=n_layers, batch_first=True)
        self.out_proj = nn.Linear(hidden_dim, n_classes_with_specials)
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

    def _init_hidden(self, encoder_outputs: torch.Tensor, encoder_mask: torch.Tensor) -> torch.Tensor:
        """h_0 = o_N in the paper (Section 4.3.2 — the LAST real encoder
        timestep's output, since the encoder is causal and o_N has
        already integrated the whole sequence by construction). Batches
        have variable real lengths, so this gathers each sample's own
        last real timestep individually rather than always index -1."""
        lengths = encoder_mask.sum(dim=1).clamp(min=1) - 1   # index of each sample's last real clip
        batch_idx = torch.arange(encoder_outputs.shape[0], device=encoder_outputs.device)
        last_real = encoder_outputs[batch_idx, lengths]        # (B, H)
        return last_real.unsqueeze(0).repeat(self.n_layers, 1, 1)   # (n_layers, B, H)

    def forward_step(self, prev_token: torch.Tensor, hidden: torch.Tensor,
                      encoder_outputs: torch.Tensor, encoder_mask: torch.Tensor):
        embedded = self.embedding(prev_token)                              # (B, embed_dim)
        context, alpha = self.attention(hidden[-1], encoder_outputs, encoder_mask)
        gru_in = torch.cat([embedded, context], dim=1).unsqueeze(1)         # (B, 1, embed_dim+H)
        out, hidden = self.gru(gru_in, hidden)
        logits = self.out_proj(out.squeeze(1))
        return logits, hidden, alpha

    def forward(self, encoder_outputs: torch.Tensor, encoder_mask: torch.Tensor,
                target_tokens: torch.Tensor = None, max_len: int = 20,
                bos_idx: int = 0, teacher_forcing: bool = True) -> torch.Tensor:
        """Training (target_tokens given, shape (B, U)): teacher-forced —
        feeds the REAL previous letter at every step regardless of what
        was predicted, matching Eq. 9. Inference (target_tokens=None):
        autoregressive — feeds back its OWN previous prediction, runs for
        max_len steps (see this file's own top-level docstring on <EOS>
        early-stopping being left to the caller)."""
        B = encoder_outputs.shape[0]
        device = encoder_outputs.device
        hidden = self._init_hidden(encoder_outputs, encoder_mask)
        prev_token = torch.full((B,), bos_idx, dtype=torch.long, device=device)
        all_logits = []
        steps = target_tokens.shape[1] if target_tokens is not None else max_len
        for t in range(steps):
            logits, hidden, _ = self.forward_step(prev_token, hidden, encoder_outputs, encoder_mask)
            all_logits.append(logits)
            prev_token = target_tokens[:, t] if (target_tokens is not None and teacher_forcing) \
                else logits.argmax(dim=-1)
        return torch.stack(all_logits, dim=1)   # (B, steps, n_classes_with_specials)


class WriteASModel(nn.Module):
    """Full replication: MultimodalCNN (per-clip) -> DilatedCausalConvEncoder
    (over the resulting clip sequence) -> {CTC head, AttentionDecoder}.
    IMU-only — see this file's own top-level docstring."""

    def __init__(self, n_classes: int, n_modalities: int = 2, clip_dim: int = 32,
                 encoder_channels: int = 128, clip_sec: float = 0.3, clip_overlap: float = 0.5,
                 sample_rate: float = None):
        super().__init__()
        self.clip_sec = clip_sec
        self.clip_overlap = clip_overlap
        self.sample_rate = sample_rate or config_ctc.IMU_RESAMPLE_HZ
        self.n_modalities = n_modalities
        self.cnn = MultimodalCNN(n_modalities=n_modalities, out_dim=clip_dim)
        self.clip_to_encoder = nn.Linear(clip_dim, encoder_channels)
        self.encoder = DilatedCausalConvEncoder(channels=encoder_channels)
        self.ctc_head = nn.Linear(encoder_channels, n_classes + 1)   # +1: blank, matches this
                                                                        # project's own BLANK_IDX=0
        # Attention decoder classes: [0..n_classes-1]=real letters,
        # n_classes=<BOS>, n_classes+1=<EOS>
        self.bos_idx = n_classes
        self.eos_idx = n_classes + 1
        self.attention_decoder = AttentionDecoder(n_classes + 2, hidden_dim=encoder_channels)

    def _split_modalities(self, imu: torch.Tensor) -> list:
        """(B, 6, T) -> [accel (B,1,T,3), gyro (B,1,T,3)] — see this
        file's own top-level docstring on why 2 branches, not the
        paper's 3 (no separate gravity channel in this project's IMU
        data)."""
        accel = imu[:, 0:3, :].transpose(1, 2).unsqueeze(1)   # (B, 1, T, 3)
        gyro = imu[:, 3:6, :].transpose(1, 2).unsqueeze(1)     # (B, 1, T, 3)
        return [accel, gyro]

    def encode(self, imu: torch.Tensor, imu_lengths: torch.Tensor):
        """imu: (B, 6, T) -> (encoder_outputs (B, n_clips, encoder_channels), clip_mask (B, n_clips),
        n_clips_per_sample (B,))."""
        clips, n_clips_per_sample = _window_clips(imu, imu_lengths, self.clip_sec, self.clip_overlap,
                                                    self.sample_rate)
        B, n_clips, C, clip_len = clips.shape
        modality_clips_flat = self._split_modalities(clips.reshape(B * n_clips, C, clip_len))
        clip_vecs = self.cnn(modality_clips_flat)                    # (B*n_clips, clip_dim)
        clip_vecs = clip_vecs.reshape(B, n_clips, -1)                  # (B, n_clips, clip_dim)
        clip_vecs = self.clip_to_encoder(clip_vecs).transpose(1, 2)     # (B, encoder_channels, n_clips)
        encoder_out = self.encoder(clip_vecs).transpose(1, 2)           # (B, n_clips, encoder_channels)
        clip_mask = (torch.arange(n_clips, device=imu.device).unsqueeze(0)
                     < n_clips_per_sample.unsqueeze(1))
        return encoder_out, clip_mask, n_clips_per_sample

    def forward(self, imu: torch.Tensor, imu_lengths: torch.Tensor,
                target_tokens: torch.Tensor = None, teacher_forcing: bool = True,
                max_decode_len: int = 20):
        """Returns (ctc_log_probs (n_clips, B, n_classes+1) — CTCLoss's
        expected time-major layout, n_clips_per_sample (B,),
        attention_logits (B, steps, n_classes+2))."""
        encoder_out, clip_mask, n_clips_per_sample = self.encode(imu, imu_lengths)
        ctc_logits = self.ctc_head(encoder_out)
        ctc_log_probs = F.log_softmax(ctc_logits, dim=2).transpose(0, 1)   # (n_clips, B, n_classes+1)
        attention_logits = self.attention_decoder(
            encoder_out, clip_mask, target_tokens=target_tokens, max_len=max_decode_len,
            bos_idx=self.bos_idx, teacher_forcing=teacher_forcing)
        return ctc_log_probs, n_clips_per_sample, attention_logits
