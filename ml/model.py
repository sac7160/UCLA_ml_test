"""
digit_classifier/model.py
────────────────────────────────────────────────────────────────────────────
Three model variants sharing the same two encoders:

  DigitFusionNet  audio + IMU, concatenated before the classifier head.
  AudioOnlyNet    audio alone — for measuring how much the IMU branch is
                   actually contributing (or isn't).
  IMUOnlyNet      IMU alone — same, from the other side.

Pick one via --modality in train.py/test.py ('fusion' / 'audio' / 'imu').
dataset.py's DigitStrokeDataset always returns both audio and IMU tensors
regardless of modality (simpler, and IMU loading is cheap next to the
audio path) — forward_model() below is the one place that decides which
of them actually get used, so train.py/test.py never need their own
if/elif on modality.
"""

import torch
import torch.nn as nn

import config

MODALITIES = ('fusion', 'audio', 'imu')


class AudioEncoder(nn.Module):
    """Input: (B, 1, N_MELS, T) log-mel spectrogram -> (B, embed_dim)."""

    def __init__(self, embed_dim: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),   # -> (B, 64, 1, 1) regardless of input T/n_mels
        )
        self.fc = nn.Linear(64, embed_dim)

    def forward(self, x):
        x = self.conv(x).flatten(1)
        return self.fc(x)


class IMUEncoder(nn.Module):
    """Input: (B, C, IMU_STEPS) resampled accel/gyro sequence -> (B, embed_dim)."""

    def __init__(self, in_channels: int = len(config.IMU_CHANNELS), embed_dim: int = 32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2), nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(64, embed_dim)

    def forward(self, x):
        x = self.conv(x).flatten(1)
        return self.fc(x)


class DigitFusionNet(nn.Module):
    def __init__(self, n_classes: int, audio_embed: int = 64, imu_embed: int = 32):
        super().__init__()
        self.audio_enc = AudioEncoder(audio_embed)
        self.imu_enc = IMUEncoder(embed_dim=imu_embed)
        self.head = nn.Sequential(
            nn.Linear(audio_embed + imu_embed, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )

    def forward(self, audio, imu):
        a = self.audio_enc(audio)
        i = self.imu_enc(imu)
        return self.head(torch.cat([a, i], dim=1))


class AudioOnlyNet(nn.Module):
    def __init__(self, n_classes: int, audio_embed: int = 64):
        super().__init__()
        self.audio_enc = AudioEncoder(audio_embed)
        self.head = nn.Sequential(
            nn.Linear(audio_embed, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )

    def forward(self, audio):
        return self.head(self.audio_enc(audio))


class IMUOnlyNet(nn.Module):
    def __init__(self, n_classes: int, imu_embed: int = 32):
        super().__init__()
        self.imu_enc = IMUEncoder(embed_dim=imu_embed)
        self.head = nn.Sequential(
            nn.Linear(imu_embed, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )

    def forward(self, imu):
        return self.head(self.imu_enc(imu))


def build_model(modality: str, n_classes: int) -> nn.Module:
    if modality == 'fusion':
        return DigitFusionNet(n_classes)
    elif modality == 'audio':
        return AudioOnlyNet(n_classes)
    elif modality == 'imu':
        return IMUOnlyNet(n_classes)
    raise ValueError(f'modality must be one of {MODALITIES}, got "{modality}"')


def forward_model(model: nn.Module, modality: str, audio: torch.Tensor, imu: torch.Tensor) -> torch.Tensor:
    """The Dataset always hands back both audio and imu tensors — this is
    the one place that decides which of them the chosen model variant
    actually consumes, so callers (run_epoch/evaluate) never need their
    own if/elif on modality."""
    if modality == 'fusion':
        return model(audio, imu)
    elif modality == 'audio':
        return model(audio)
    elif modality == 'imu':
        return model(imu)
    raise ValueError(f'modality must be one of {MODALITIES}, got "{modality}"')