"""
dataset_ctc_concat.py
────────────────────────────────────────────────────────────────────────────
Synthesizes multi-letter "fake word" training examples by concatenating
several single-letter trials end-to-end, with a short silence/rest gap
between each — see the chat this was added in for the exact symptom this
fixes: a LetterCTCNet trained ONLY on length-1 targets never gets a
training signal that says "keep emitting different letters as time goes
on" — every single training example was satisfied by "spike ONE letter,
blank everywhere else", which is exactly what it learns, and exactly why
it collapses to a single character (mean predicted length ~1) on much
longer real word trials at inference time, regardless of how long or
letter-rich those trials actually are.

This gives the model real multi-letter targets to train on, synthesized
entirely from data already collected — no new recording needed. It's a
standard technique for bridging from isolated-character training data to
a model that needs to work on continuous multi-character input (the same
idea shows up in OCR/handwriting-recognition literature under "synthetic
line generation" from isolated character/glyph datasets).

Train on a MIX of LetterDatasetCTC (real, single-letter) and
ConcatLetterDatasetCTC (synthetic, multi-letter) — see train_ctc.py's
--synthetic-per-epoch / --concat-letters-min / --concat-letters-max.
dataset_ctc.py itself is untouched; this file only reads its already-
tested _resample_to_rate/_MEL/_DB and duplicates the small "raw, pre-
normalization" loading step those functions don't expose on their own.
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio
from scipy.io import wavfile
from torch.utils.data import Dataset

import config_ctc
from dataset_ctc import (_MEL, _DB, _resample_to_rate, _smooth, _robust_origin_normalize,
                          _denoise_fingertip_signal)

GAP_SEC = 0.15   # silence (audio) / held-position (IMU) inserted between concatenated letters —
                  # loosely approximates the brief pause a real finger-lift between letters produces;
                  # not tuned against real word-trial gap statistics, just a reasonable starting guess


def _load_raw_waveform(trial_dir: Path, audio_source: str) -> torch.Tensor:
    """(1, N) waveform resampled to config_ctc.AUDIO_TARGET_SR — everything
    load_audio_variable() does BEFORE the mel-spectrogram/normalization
    steps, factored out here so several of these can be concatenated
    first and the mel transform applied ONCE on the full concatenated
    signal (matching how a real trial's audio is processed — never
    per-segment), rather than concatenating already-computed spectrograms
    (which would paste together mismatched/independently-normalized
    frequency content at the seams)."""
    wav_path = trial_dir / config_ctc.AUDIO_FILENAMES[audio_source]
    sr, samples = wavfile.read(wav_path)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if np.issubdtype(samples.dtype, np.integer):
        samples = samples.astype(np.float32) / np.iinfo(samples.dtype).max
    else:
        samples = samples.astype(np.float32)
    waveform = torch.from_numpy(samples).unsqueeze(0)
    if sr != config_ctc.AUDIO_TARGET_SR:
        waveform = torchaudio.functional.resample(waveform, sr, config_ctc.AUDIO_TARGET_SR)
    return waveform


def _load_raw_imu(trial_dir: Path, imu_source: str, finger: str, denoise: bool = True) -> "torch.Tensor | None":
    """(6, T) accel+gyro, resampled at config_ctc.IMU_RESAMPLE_HZ — same
    idea as _load_raw_waveform above: everything load_imu_variable() does
    BEFORE z-normalization, so several of these can be concatenated first
    and normalized ONCE over the full concatenated sequence. Returns None
    if the source trial doesn't have enough real samples to resample from
    (mirrors load_imu_variable's own all-zero fallback, just deferred to
    the caller here since "zeros of what length" depends on context this
    function doesn't have)."""
    if imu_source == 'fingertip':
        df = pd.read_csv(trial_dir / config_ctc.IMU_FILENAMES['fingertip'])
        df = df[(df['finger'] == finger) & (df['detected'])]
        if len(df) < 2:
            return None
        t = df['time_aligned'].to_numpy()
        values = df[config_ctc.IMU_CHANNELS].to_numpy()
        imu = _resample_to_rate(t, values, config_ctc.IMU_RESAMPLE_HZ, config_ctc.IMU_MAX_STEPS_CAP)
    else:
        df = pd.read_csv(trial_dir / config_ctc.IMU_FILENAMES['watch'])
        acc = df[df['sensor'] == 'acc']
        gyro = df[df['sensor'] == 'gyro']
        if len(acc) < 2 or len(gyro) < 2:
            return None
        acc_r = _resample_to_rate(acc['time_aligned'].to_numpy(), acc[['v1', 'v2', 'v3']].to_numpy(),
                                   config_ctc.IMU_RESAMPLE_HZ, config_ctc.IMU_MAX_STEPS_CAP)
        gyro_r = _resample_to_rate(gyro['time_aligned'].to_numpy(), gyro[['v1', 'v2', 'v3']].to_numpy(),
                                    config_ctc.IMU_RESAMPLE_HZ, config_ctc.IMU_MAX_STEPS_CAP)
        n = min(acc_r.shape[1], gyro_r.shape[1])
        imu = np.concatenate([acc_r[:, :n], gyro_r[:, :n]], axis=0)
    imu = imu.astype(np.float32)
    if imu_source == 'fingertip' and denoise:
        # Per-segment, before concatenation — same "denoise regardless of
        # role" reasoning as dataset_ctc.load_imu_variable: MediaPipe
        # tracking jitter is a property of the source, whether this pick
        # ends up used as the model's input or as the IMU decoder's
        # teacher target. See dataset_ctc._denoise_fingertip_signal.
        imu = _denoise_fingertip_signal(imu)
    # Smoothed HERE, per-segment, before concatenation with other picks —
    # smoothing AFTER concatenation would blur real motion across the
    # artificial gap between two different letters, which is exactly the
    # kind of distortion smoothing is meant to avoid introducing.
    smooth_window = max(1, int(config_ctc.IMU_SMOOTH_WINDOW_SEC * config_ctc.IMU_RESAMPLE_HZ))
    imu = _smooth(imu, smooth_window)
    return torch.from_numpy(imu.astype(np.float32))




class ConcatLetterDatasetCTC(Dataset):
    """Each __getitem__ call SYNTHESIZES one fresh multi-letter example —
    picks n_letters random (trial_dir, label) pairs from `samples`
    (typically the same train split LetterDatasetCTC uses — never val/
    test, to avoid leaking those trials' content into training even in
    concatenated form), concatenates their audio/IMU with GAP_SEC of
    silence/held-position between each, and returns a target sequence of
    all n_letters labels in the order concatenated.

    len() is n_synthetic_per_epoch, a number YOU choose (not derived from
    `samples`) — since every __getitem__ call generates a new random
    combination, this dataset has no fixed "real" size; it defines how
    many synthetic examples one epoch sees, same role len(train_ds) plays
    for the real, non-synthetic data.

    Output shapes match LetterDatasetCTC exactly (mel, imu, target) —
    dataset_ctc.py's collate_fn_ctc works on this unmodified; the two can
    be combined with torch.utils.data.ConcatDataset directly."""

    def __init__(self, samples: list, n_synthetic_per_epoch: int,
                 n_letters_range: tuple = (2, 5),
                 finger: str = config_ctc.DEFAULT_FINGER,
                 audio_source: str = config_ctc.DEFAULT_AUDIO_SOURCE,
                 imu_source: str = config_ctc.DEFAULT_IMU_SOURCE,
                 seed: int = None, denoise_fingertip: bool = True):
        if not samples:
            raise RuntimeError('ConcatLetterDatasetCTC got an empty sample list')
        self.samples = samples
        self.n_synthetic_per_epoch = n_synthetic_per_epoch
        self.n_letters_range = n_letters_range
        self.finger = finger
        self.audio_source = audio_source
        self.imu_source = imu_source
        self.denoise_fingertip = denoise_fingertip
        self.rng = random.Random(seed)

    def __len__(self):
        return self.n_synthetic_per_epoch

    def __getitem__(self, idx):
        n = self.rng.randint(*self.n_letters_range)
        candidates = [self.rng.choice(self.samples) for _ in range(n)]

        # Decide which picks are actually usable BEFORE building either
        # audio or IMU — audio/imu/target all have to agree on the exact
        # same letter sequence, so a pick that fails to load for IMU
        # can't just be silently dropped from IMU alone while audio still
        # includes it (that used to be a real bug here: audio built from
        # ALL n picks, IMU/target only from the ones that survived IMU
        # loading, so a fusion-modality sample could end up with a
        # 4-letter audio span paired against a 3-letter target). IMU is
        # loaded here once and reused below rather than reloaded, since
        # that load is exactly the check being performed. The IMU
        # decoder's reconstruction TARGET is loaded separately here too
        # — always the 'fingertip' source regardless of self.imu_source,
        # same picks, mirroring how surface_wave_pieces below is always
        # the 'surface' audio source regardless of self.audio_source.
        picks = []
        imu_segs = []
        imu_target_segs = []
        for trial_dir, label in candidates:
            seg = _load_raw_imu(trial_dir, self.imu_source, self.finger, denoise=self.denoise_fingertip)
            if seg is None:
                continue
            target_seg = _load_raw_imu(trial_dir, 'fingertip', self.finger, denoise=self.denoise_fingertip)
            if target_seg is None:
                continue
            picks.append((trial_dir, label))
            imu_segs.append(seg)
            imu_target_segs.append(target_seg)
        if not picks:
            picks = [candidates[0]]   # extremely unlikely across multiple random draws — fall back
                                       # to a single letter (accepting it may be IMU-unusable) rather
                                       # than returning an empty sample
            imu_segs = [torch.zeros(len(config_ctc.IMU_CHANNELS), 2, dtype=torch.float32)]
            imu_target_segs = [torch.zeros(len(config_ctc.IMU_CHANNELS), 2, dtype=torch.float32)]

        gap_samples = int(GAP_SEC * config_ctc.AUDIO_TARGET_SR)
        wave_pieces = []
        surface_wave_pieces = []   # same picks, same gap pattern, but always the 'surface' source —
                                     # the Mic decoder's reconstruction target, mirroring how
                                     # imu_target_segs above is always fingertip regardless of self.imu_source
        for trial_dir, _ in picks:
            wave_pieces.append(_load_raw_waveform(trial_dir, self.audio_source))
            wave_pieces.append(torch.zeros(1, gap_samples))
            surface_wave_pieces.append(_load_raw_waveform(trial_dir, 'surface'))
            surface_wave_pieces.append(torch.zeros(1, gap_samples))
        full_wave = torch.cat(wave_pieces[:-1], dim=1)      # drop the trailing gap
        max_len = int(config_ctc.AUDIO_MAX_SEC_CAP * config_ctc.AUDIO_TARGET_SR)
        if full_wave.shape[1] > max_len:
            full_wave = full_wave[:, :max_len]
        mel = _DB(_MEL(full_wave))
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)

        full_surface_wave = torch.cat(surface_wave_pieces[:-1], dim=1)
        if full_surface_wave.shape[1] > max_len:
            full_surface_wave = full_surface_wave[:, :max_len]
        surface_mel = _DB(_MEL(full_surface_wave))
        surface_mel = (surface_mel - surface_mel.mean()) / (surface_mel.std() + 1e-6)

        gap_steps = max(1, int(GAP_SEC * config_ctc.IMU_RESAMPLE_HZ))
        imu_pieces = []
        for seg in imu_segs:
            imu_pieces.append(seg)
            imu_pieces.append(seg[:, -1:].repeat(1, gap_steps))   # hold last position — a
                                                                    # literal-zero gap would look like
                                                                    # an unrealistic snap to the origin
        full_imu = torch.cat(imu_pieces[:-1], dim=1)
        mean = full_imu.mean(dim=1, keepdim=True)
        std = full_imu.std(dim=1, keepdim=True) + 1e-6
        full_imu = (full_imu - mean) / std

        imu_target_pieces = []
        for seg in imu_target_segs:
            imu_target_pieces.append(seg)
            imu_target_pieces.append(seg[:, -1:].repeat(1, gap_steps))
        full_imu_target = torch.cat(imu_target_pieces[:-1], dim=1)
        # Robust local origin, NOT z-score — see dataset_ctc._robust_origin_normalize's
        # docstring for why: z-score's std can be tiny during a near-
        # stationary stretch, so dividing by it would blow small camera-
        # tracking jitter up to look like large, meaningful movement,
        # exactly the wrong behavior for a decoder's training TARGET.
        # full_imu (the model's actual INPUT, built above) intentionally
        # keeps z-score — this only changes the teacher signal.
        full_imu_target = torch.from_numpy(_robust_origin_normalize(full_imu_target.numpy()))

        target = torch.tensor([label + 1 for _, label in picks], dtype=torch.long)
        return mel, full_imu, target, full_imu_target, surface_mel
