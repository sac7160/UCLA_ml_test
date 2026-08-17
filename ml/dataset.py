"""
digit_classifier/dataset.py
────────────────────────────────────────────────────────────────────────────
Three separate responsibilities, kept apart on purpose:

  scan_dataset()  walks dataset_root/<class>/trial_*/ and returns every
                   usable (trial_dir, label) pair — no splitting yet.
  make_splits()    takes that full list and does a *stratified* (per-class)
                   train/val/test split, so a class isn't accidentally
                   starved out of one split by chance.
  save_splits() /
  load_splits()    persist the split (plus which modality sources/finger
                   were used) to JSON, so train.py and test.py are
                   guaranteed to agree on both exactly which trials are in
                   the test set, AND how to read them — the test set must
                   never be re-derived from scratch, or it silently stops
                   being held-out.

DigitStrokeDataset itself takes an explicit sample list (not a directory to
scan) — train.py builds three instances of it, one per split, all sharing
this same preprocessing code.

Two independent choices decide what a trial's input actually is:
  --audio-source  'surface' (surface_mic.wav) or 'watch' (watch_audio.wav)
  --imu-source    'fingertip' (fingertip_imu.csv) or 'watch' (imu.csv)
Both audio sources are plain wav files, read identically. The two IMU
sources have genuinely different CSV schemas (see _load_imu_fingertip vs
_load_imu_watch below) but produce the same (6, IMU_STEPS) shape either
way, so model.py's IMUEncoder doesn't need to know or care which was used.

PairedStrokeDataset (for pretrain_cross_modal.py) returns all four signals
for the same trial at once — the watch pair and the surface+fingertip
"teacher" pair — sharing the exact same load_audio()/load_imu() functions
DigitStrokeDataset uses, so there's no risk of the two ever preprocessing
a signal differently.
"""

import json
import random
import wave
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
from scipy.io import wavfile
from torch.utils.data import Dataset

import config


# ─── Scanning & splitting ──────────────────────────────────────────────────────
def _wav_is_valid(path: Path) -> bool:
    """A failed/interrupted recording (e.g. the watch disconnected mid-trial)
    can leave a 0-frame wav file behind — it exists, so a plain
    exists()-check lets it through, but there's nothing to resample and
    torchaudio crashes trying. Cheap header-only read, not a full decode."""
    try:
        with wave.open(str(path), 'rb') as wf:
            return wf.getnframes() > 0
    except Exception:
        return False


def _csv_has_data_rows(path: Path) -> bool:
    """Same idea as _wav_is_valid, for the IMU CSV — a header line with
    nothing after it is a completed-but-empty file, not a missing one."""
    try:
        with open(path) as f:
            next(f)              # header
            return next(f, None) is not None
    except Exception:
        return False


def scan_dataset(dataset_root: Path, classes: list,
                  audio_source: str = config.DEFAULT_AUDIO_SOURCE,
                  imu_source: str = config.DEFAULT_IMU_SOURCE) -> list:
    """Returns [(trial_dir: Path, label: int), ...] for every trial under
    dataset_root/<class>/trial_*/ that has both required files for the
    chosen sources, *and* both files actually contain data — a trial from
    a failed/interrupted recording can have the right files present but
    empty, which passing that trial through would only crash on later (see
    _wav_is_valid/_csv_has_data_rows) or silently degrade training (an
    empty IMU file already falls back to an all-zero input in _load_imu,
    but a real trial that happens to be unrecoverable shouldn't be in the
    dataset teaching the model that this class "looks like" all zeros).
    `label` is the trial's index into `classes`."""
    audio_filename = config.AUDIO_FILENAMES[audio_source]
    imu_filename = config.IMU_FILENAMES[imu_source]

    samples = []
    for label, cls in enumerate(classes):
        cls_dir = dataset_root / cls
        if not cls_dir.exists():
            print(f'[WARN] no folder for class "{cls}" under {dataset_root} — skipping it entirely')
            continue
        trial_dirs = sorted(p for p in cls_dir.glob('trial_*') if p.is_dir())
        n_ok = 0
        for trial_dir in trial_dirs:
            audio_path = trial_dir / audio_filename
            imu_path = trial_dir / imu_filename
            if not audio_path.exists() or not imu_path.exists():
                print(f'[WARN] {trial_dir} missing {audio_filename} or {imu_filename} — skipping')
                continue
            if not _wav_is_valid(audio_path):
                print(f'[WARN] {trial_dir}: {audio_filename} has 0 audio frames — skipping '
                      f'(likely an interrupted/failed recording)')
                continue
            if not _csv_has_data_rows(imu_path):
                print(f'[WARN] {trial_dir}: {imu_filename} has no data rows — skipping '
                      f'(likely an interrupted/failed recording)')
                continue
            samples.append((trial_dir, label))
            n_ok += 1
        print(f'[DATA] {cls}: {n_ok}/{len(trial_dirs)} trials usable '
              f'(audio_source="{audio_source}", imu_source="{imu_source}")')

    if not samples:
        raise RuntimeError(f'No usable trials found under {dataset_root} for classes {classes} '
                            f'with audio_source="{audio_source}", imu_source="{imu_source}"')
    return samples


def scan_paired_dataset(dataset_root: Path, classes: list, finger: str = config.DEFAULT_FINGER) -> list:
    """Like scan_dataset(), but for cross-modal pretraining — a trial only
    counts if ALL FOUR files (surface_mic.wav, watch_audio.wav,
    fingertip_imu.csv, imu.csv) are present and non-empty, since
    pretrain_cross_modal.py needs the watch pair AND the surface+fingertip
    pair from the exact same trial simultaneously."""
    samples = []
    for label, cls in enumerate(classes):
        cls_dir = dataset_root / cls
        if not cls_dir.exists():
            print(f'[WARN] no folder for class "{cls}" under {dataset_root} — skipping it entirely')
            continue
        trial_dirs = sorted(p for p in cls_dir.glob('trial_*') if p.is_dir())
        n_ok = 0
        for trial_dir in trial_dirs:
            paths = {
                'surface_mic.wav': trial_dir / config.AUDIO_FILENAMES['surface'],
                'watch_audio.wav': trial_dir / config.AUDIO_FILENAMES['watch'],
                'fingertip_imu.csv': trial_dir / config.IMU_FILENAMES['fingertip'],
                'imu.csv': trial_dir / config.IMU_FILENAMES['watch'],
            }
            missing = [name for name, p in paths.items() if not p.exists()]
            if missing:
                print(f'[WARN] {trial_dir} missing {", ".join(missing)} — skipping')
                continue
            if not (_wav_is_valid(paths['surface_mic.wav']) and _wav_is_valid(paths['watch_audio.wav'])):
                print(f'[WARN] {trial_dir}: an audio file has 0 frames — skipping '
                      f'(likely an interrupted/failed recording)')
                continue
            if not (_csv_has_data_rows(paths['fingertip_imu.csv']) and _csv_has_data_rows(paths['imu.csv'])):
                print(f'[WARN] {trial_dir}: an IMU file has no data rows — skipping '
                      f'(likely an interrupted/failed recording)')
                continue
            samples.append((trial_dir, label))
            n_ok += 1
        print(f'[DATA] {cls}: {n_ok}/{len(trial_dirs)} trials usable for paired (watch + surface+fingertip) loading')

    if not samples:
        raise RuntimeError(f'No usable paired trials found under {dataset_root} for classes {classes}')
    return samples


def make_splits(samples: list, train_frac: float, val_frac: float, test_frac: float, seed: int) -> dict:
    """Stratified train/val/test split — each class is shuffled and cut
    independently, so every class contributes to all three splits in
    roughly the same proportion (a single global shuffle risks a class
    landing almost entirely in one split by chance, especially with only
    a few dozen trials per class in a pilot like this).

    For a class with very few trials, val/test are shrunk (never below 0)
    so at least one example is left for training — with a warning, since
    that class's val/test metrics will still be noisy either way."""
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, \
        'train_frac + val_frac + test_frac must sum to 1.0'

    by_class = {}
    for trial_dir, label in samples:
        by_class.setdefault(label, []).append(trial_dir)

    rng = random.Random(seed)
    splits = {'train': [], 'val': [], 'test': []}
    for label, dirs in sorted(by_class.items()):
        dirs = dirs[:]
        rng.shuffle(dirs)
        n = len(dirs)
        n_test = max(1, round(n * test_frac))
        n_val = max(1, round(n * val_frac))
        if n_test + n_val >= n:
            n_test = min(n_test, max(0, n - 2))
            n_val = min(n_val, max(0, n - n_test - 1))
        if n - n_test - n_val <= 0:
            print(f'[WARN] class label={label} has only {n} trial(s) — too few for a proper '
                  f'train/val/test split; results for this class will be unreliable')

        test_dirs = dirs[:n_test]
        val_dirs = dirs[n_test:n_test + n_val]
        train_dirs = dirs[n_test + n_val:]
        splits['test'].extend((d, label) for d in test_dirs)
        splits['val'].extend((d, label) for d in val_dirs)
        splits['train'].extend((d, label) for d in train_dirs)

    for name in splits:
        rng.shuffle(splits[name])
    return splits


def save_splits(splits: dict, path: Path, classes: list, finger: str, seed: int,
                 audio_source: str = config.DEFAULT_AUDIO_SOURCE,
                 imu_source: str = config.DEFAULT_IMU_SOURCE):
    payload = {
        'classes': classes,
        'finger': finger,
        'audio_source': audio_source,
        'imu_source': imu_source,
        'seed': seed,
        'splits': {
            name: [[str(trial_dir), label] for trial_dir, label in pairs]
            for name, pairs in splits.items()
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)


def load_splits(path: Path):
    """Returns (splits, classes, finger, audio_source, imu_source, seed) —
    splits[name] is a list of (Path, label) pairs, same shape make_splits()
    produces. Splits files saved before audio_source/imu_source existed
    fall back to the old surface+fingertip defaults."""
    with open(path) as f:
        payload = json.load(f)
    splits = {
        name: [(Path(trial_dir), label) for trial_dir, label in pairs]
        for name, pairs in payload['splits'].items()
    }
    audio_source = payload.get('audio_source', config.DEFAULT_AUDIO_SOURCE)
    imu_source = payload.get('imu_source', config.DEFAULT_IMU_SOURCE)
    return splits, payload['classes'], payload['finger'], audio_source, imu_source, payload['seed']


# ─── Shared resampling helper ──────────────────────────────────────────────────
def _resample_to_fixed_steps(t: np.ndarray, values: np.ndarray, steps: int) -> np.ndarray:
    """values: (N, C) sampled at times t (any units, not required to be
    sorted or evenly spaced). Returns (C, steps), linearly interpolated
    over duration-normalized time (0..1) — used by both IMU loaders below,
    since fingertip and watch IMU differ in how the raw CSV is shaped but
    both ultimately need "N irregular samples -> fixed-length sequence"."""
    order = np.argsort(t)
    t_sorted = t[order]
    values_sorted = values[order]
    t_norm = (t_sorted - t_sorted[0]) / max(t_sorted[-1] - t_sorted[0], 1e-6)
    t_query = np.linspace(0.0, 1.0, steps)
    out = np.empty((values.shape[1], steps), dtype=np.float32)
    for c in range(values.shape[1]):
        out[c] = np.interp(t_query, t_norm, values_sorted[:, c])
    return out


# ─── Shared per-trial loaders (module-level, not bound to any Dataset —
# both DigitStrokeDataset below and PairedStrokeDataset in
# paired_dataset.py call these directly, so the two datasets can never
# silently drift apart in how they read the same files) ──────────────────────
_MEL = torchaudio.transforms.MelSpectrogram(
    sample_rate=config.AUDIO_TARGET_SR, n_fft=config.N_FFT,
    hop_length=config.HOP_LENGTH, n_mels=config.N_MELS,
)
_DB = torchaudio.transforms.AmplitudeToDB(stype='power')


def load_audio(trial_dir: Path, audio_source: str) -> torch.Tensor:
    # scipy.io.wavfile rather than torchaudio.load(): the same reader the
    # data-collection app already uses, and it avoids torchaudio's I/O
    # backend needing a separate torchcodec install on top — torchaudio
    # is still used below for resample/mel-spectrogram, which are pure
    # tensor ops with no separate backend dependency.
    wav_path = trial_dir / config.AUDIO_FILENAMES[audio_source]
    sr, samples = wavfile.read(wav_path)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if np.issubdtype(samples.dtype, np.integer):
        samples = samples.astype(np.float32) / np.iinfo(samples.dtype).max
    else:
        samples = samples.astype(np.float32)
    waveform = torch.from_numpy(samples).unsqueeze(0)   # (1, N)

    if sr != config.AUDIO_TARGET_SR:
        waveform = torchaudio.functional.resample(waveform, sr, config.AUDIO_TARGET_SR)

    target_len = int(config.AUDIO_MAX_SEC * config.AUDIO_TARGET_SR)
    if waveform.shape[1] < target_len:
        waveform = F.pad(waveform, (0, target_len - waveform.shape[1]))
    else:
        waveform = waveform[:, :target_len]

    mel = _DB(_MEL(waveform))   # (1, N_MELS, T)
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)   # per-sample normalization
    return mel


def _downsample_to_rate(t: np.ndarray, values: np.ndarray, target_hz: float):
    """Simulates a slower physical sensor by picking the nearest REAL
    sample at each point on a coarser time grid — not interpolation.
    A slow-frame-rate camera doesn't get to peek at fast in-between
    motion and average it away; it simply never captures it. Nearest-
    neighbor selection is what actually reproduces that: interpolating
    from the full-rate signal first and then discarding points would
    still leak fine temporal detail into the "downsampled" result that a
    genuinely slower sensor would never have recorded."""
    t_target = np.arange(t[0], t[-1], 1.0 / target_hz)
    if len(t_target) < 2:
        t_target = t[[0, -1]]
    idx = np.clip(np.searchsorted(t, t_target), 0, len(t) - 1)
    return t_target, values[idx]


def _load_imu_fingertip(trial_dir: Path, finger: str) -> np.ndarray | None:
    df = pd.read_csv(trial_dir / config.IMU_FILENAMES['fingertip'])
    df = df[(df['finger'] == finger) & (df['detected'])]
    if len(df) < 2:
        print(f'[WARN] {trial_dir}: only {len(df)} detected samples for finger="{finger}" '
              f'— using an all-zero IMU input for this trial')
        return None
    t = df['time_aligned'].to_numpy()
    values = df[config.IMU_CHANNELS].to_numpy()   # already 6 columns, already paired per-row
    return _resample_to_fixed_steps(t, values, config.IMU_STEPS)


def _load_imu_fingertip_pos_vel(trial_dir: Path, finger: str) -> np.ndarray | None:
    """Position + velocity (ONE finite-difference derivative) instead of
    the accel/gyro columns (which fingertip_imu_multi.py computed via TWO
    derivatives of the same noisy camera-tracked position). Ablation for
    testing whether it's the double-differentiation that's amplifying
    tracking noise into a much worse signal, rather than the camera-based
    tracking itself being fundamentally worse than the watch's real
    accelerometer/gyro."""
    df = pd.read_csv(trial_dir / config.IMU_FILENAMES['fingertip_position'])
    df = df[(df['finger'] == finger) & (df['detected'])]
    if len(df) < 3:
        print(f'[WARN] {trial_dir}: only {len(df)} detected samples for finger="{finger}" '
              f'— using an all-zero IMU input for this trial')
        return None
    df = df.sort_values('time_aligned')
    t = df['time_aligned'].to_numpy()
    pos = df[['pos_x', 'pos_y', 'pos_z']].to_numpy()
    vel = np.gradient(pos, t, axis=0)   # single derivative, vs. accel's two
    combined = np.concatenate([pos, vel], axis=1)   # (N, 6): pos_x/y/z, vel_x/y/z
    return _resample_to_fixed_steps(t, combined, config.IMU_STEPS)


def _load_imu_watch(trial_dir: Path, downsample_hz: float | None = None) -> np.ndarray | None:
    # acc and gyro are separate rows with their own timestamps here, not
    # paired columns like fingertip_imu.csv, so each needs its own resample.
    df = pd.read_csv(trial_dir / config.IMU_FILENAMES['watch'])
    acc = df[df['sensor'] == 'acc']
    gyro = df[df['sensor'] == 'gyro']
    if len(acc) < 2 or len(gyro) < 2:
        print(f'[WARN] {trial_dir}: only {len(acc)} acc / {len(gyro)} gyro samples in watch imu.csv '
              f'— using an all-zero IMU input for this trial')
        return None
    acc_t, acc_v = acc['time_aligned'].to_numpy(), acc[['v1', 'v2', 'v3']].to_numpy()
    gyro_t, gyro_v = gyro['time_aligned'].to_numpy(), gyro[['v1', 'v2', 'v3']].to_numpy()
    if downsample_hz is not None:
        # --imu-downsample-hz ablation: simulate the watch's own sensor
        # having only sampled at fingertip's native rate, to test whether
        # temporal resolution (not noise) explains the accuracy gap —
        # this happens BEFORE the fixed-64-step resample below, same as
        # the real (never-downsampled) path, so the two are otherwise
        # identical and only differ in how much real motion was captured.
        acc_t, acc_v = _downsample_to_rate(acc_t, acc_v, downsample_hz)
        gyro_t, gyro_v = _downsample_to_rate(gyro_t, gyro_v, downsample_hz)
    acc_r = _resample_to_fixed_steps(acc_t, acc_v, config.IMU_STEPS)
    gyro_r = _resample_to_fixed_steps(gyro_t, gyro_v, config.IMU_STEPS)
    return np.concatenate([acc_r, gyro_r], axis=0)   # (6, IMU_STEPS) — accel first, then gyro


def load_imu(trial_dir: Path, imu_source: str, finger: str = config.DEFAULT_FINGER,
             watch_downsample_hz: float | None = None) -> torch.Tensor:
    if imu_source == 'fingertip':
        imu = _load_imu_fingertip(trial_dir, finger)
    elif imu_source == 'fingertip_position':
        imu = _load_imu_fingertip_pos_vel(trial_dir, finger)
    else:
        imu = _load_imu_watch(trial_dir, downsample_hz=watch_downsample_hz)

    if imu is None:
        return torch.zeros(len(config.IMU_CHANNELS), config.IMU_STEPS, dtype=torch.float32)

    imu = imu.astype(np.float32)
    mean = imu.mean(axis=1, keepdims=True)
    std = imu.std(axis=1, keepdims=True) + 1e-6
    imu = (imu - mean) / std
    return torch.from_numpy(imu)


# ─── PyTorch Dataset ────────────────────────────────────────────────────────────
class DigitStrokeDataset(Dataset):
    """Wraps an explicit list of (trial_dir, label) pairs — get one from
    make_splits()/load_splits(), not by pointing this at a directory
    directly, so train/val/test each only ever see their own samples."""

    def __init__(self, samples: list, finger: str = config.DEFAULT_FINGER,
                 audio_source: str = config.DEFAULT_AUDIO_SOURCE,
                 imu_source: str = config.DEFAULT_IMU_SOURCE,
                 watch_downsample_hz: float | None = None):
        if not samples:
            raise RuntimeError('DigitStrokeDataset got an empty sample list')
        if imu_source not in config.IMU_FILENAMES:
            raise ValueError(f'imu_source must be one of {list(config.IMU_FILENAMES)}, got "{imu_source}"')
        if audio_source not in config.AUDIO_FILENAMES:
            raise ValueError(f'audio_source must be one of {list(config.AUDIO_FILENAMES)}, got "{audio_source}"')
        if watch_downsample_hz is not None and imu_source != 'watch':
            raise ValueError('watch_downsample_hz only applies when imu_source="watch"')

        self.samples = samples
        self.finger = finger
        self.audio_source = audio_source
        self.imu_source = imu_source
        self.watch_downsample_hz = watch_downsample_hz

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        trial_dir, label = self.samples[idx]
        audio = load_audio(trial_dir, self.audio_source)
        imu = load_imu(trial_dir, self.imu_source, self.finger, watch_downsample_hz=self.watch_downsample_hz)
        return audio, imu, label


class PairedStrokeDataset(Dataset):
    """For cross-modal pretraining: returns BOTH the watch pair
    (watch_audio, watch_imu) AND the surface+fingertip "teacher" pair
    (surface_audio, fingertip_imu) for the same trial, plus its label.
    Use scan_paired_dataset() to build the sample list — a plain
    scan_dataset() list isn't guaranteed to have all four files."""

    def __init__(self, samples: list, finger: str = config.DEFAULT_FINGER):
        if not samples:
            raise RuntimeError('PairedStrokeDataset got an empty sample list')
        self.samples = samples
        self.finger = finger

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        trial_dir, label = self.samples[idx]
        watch_audio = load_audio(trial_dir, 'watch')
        watch_imu = load_imu(trial_dir, 'watch')
        surface_audio = load_audio(trial_dir, 'surface')
        fingertip_imu = load_imu(trial_dir, 'fingertip', self.finger)
        return watch_audio, watch_imu, surface_audio, fingertip_imu, label