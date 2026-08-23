"""
dataset_ctc.py
────────────────────────────────────────────────────────────────────────────
Loads trials WITHOUT resampling them to a fixed step count / fixed
duration the way dataset.py's load_audio()/load_imu() do — see
config_ctc.py's docstring for why: CTC needs the time axis to keep
reflecting real relative duration, and needs to be able to load a whole,
un-segmented word span (first touch_on to last touch_off), which can be
several times longer than any single-letter trial.

LetterDatasetCTC is what train_ctc.py trains on — ordinary single-letter
trials (dataset.py's scan_dataset()/make_splits()/save_splits()/
load_splits() are reused as-is for finding/splitting them; only how each
trial's audio/IMU gets loaded is different here). Each sample's target is
a length-1 label sequence (just that one letter) — CTC doesn't need
per-timestep ground truth, only "this whole signal ultimately spells X".

evaluate_word_ctc.py (separate file) uses this module's load_audio_variable/
load_imu_variable directly on whole word trials at inference time — it
doesn't need a Dataset/DataLoader wrapper since it processes one word
trial at a time.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
from scipy.io import wavfile
from torch.utils.data import Dataset

import config_ctc

_MEL = torchaudio.transforms.MelSpectrogram(
    sample_rate=config_ctc.AUDIO_TARGET_SR, n_fft=config_ctc.N_FFT,
    hop_length=config_ctc.HOP_LENGTH, n_mels=config_ctc.N_MELS,
)
_DB = torchaudio.transforms.AmplitudeToDB(stype='power')


def load_audio_variable(trial_dir: Path, audio_source: str,
                         t_start: float = None, t_end: float = None) -> torch.Tensor:
    """(1, N_MELS, T) log-mel spectrogram, T proportional to the actual
    (optionally time-cropped) duration — never padded/truncated to a
    fixed length here; that only happens per-batch, in collate_fn_ctc,
    since different trials/batches can need different amounts of it."""
    wav_path = trial_dir / config_ctc.AUDIO_FILENAMES[audio_source]
    sr, samples = wavfile.read(wav_path)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if t_start is not None or t_end is not None:
        i0 = max(0, int((t_start or 0.0) * sr))
        i1 = min(len(samples), int((t_end if t_end is not None else len(samples) / sr) * sr))
        samples = samples[i0:i1] if i1 > i0 else samples[i0:i0 + 1]
    if len(samples) < config_ctc.N_FFT:
        # Several different ways to end up here, all handled the same
        # way: (1) the source wav itself has 0 frames — a failed/
        # interrupted recording that EXISTS but never captured anything
        # (see dataset_ctc_realtext.scan_text_trials()'s _wav_is_valid()
        # check, which catches this case before training ever gets here
        # — this is the belt-and-suspenders backstop for any OTHER caller
        # that doesn't check first); (2) the wav has SOME frames, but a
        # PARTIAL/truncated recording ended before the [t_start, t_end]
        # window events.csv asked for — i0 lands past the end of
        # `samples` entirely, so even the "at least 1 sample" fallback
        # above (samples[i0:i0+1]) returns empty, since slicing past an
        # array's end yields empty rather than wrapping or erroring; or
        # (3) the crop lands on a genuinely tiny sliver of real audio —
        # nonzero length, but still too short for the mel transform's
        # STFT, which pads by N_FFT//2 on each side internally and raises
        # if the input is shorter than that padding (a plain
        # `len(samples) == 0` check, this module's first attempt at this
        # fix, caught (1)/(2) but not this). See the chat this was caught
        # in for the exact crash each of these produced, all on real
        # word/sentence trials whose actual recorded audio turned out
        # shorter than their own touch-event timestamps implied.
        # Either way: torchaudio.resample() on a too-short/empty tensor
        # raises rather than returning empty, so this substitutes a short
        # silence instead of ever reaching that call. 0.5s is arbitrary
        # but comfortably above the N_FFT floor above; a fully healthy
        # trial should never actually hit this path regardless of the
        # exact value.
        samples = np.zeros(int(0.5 * sr), dtype=np.float32)

    if np.issubdtype(samples.dtype, np.integer):
        samples = samples.astype(np.float32) / np.iinfo(samples.dtype).max
    else:
        samples = samples.astype(np.float32)
    waveform = torch.from_numpy(samples).unsqueeze(0)

    if sr != config_ctc.AUDIO_TARGET_SR:
        waveform = torchaudio.functional.resample(waveform, sr, config_ctc.AUDIO_TARGET_SR)

    max_len = int(config_ctc.AUDIO_MAX_SEC_CAP * config_ctc.AUDIO_TARGET_SR)
    if waveform.shape[1] > max_len:
        waveform = waveform[:, :max_len]   # safety cap only — not a target length, see module docstring

    mel = _DB(_MEL(waveform))
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)
    return mel   # (1, N_MELS, T) — T left as whatever the mel transform naturally produced


def _resample_to_rate(t: np.ndarray, values: np.ndarray, target_hz: float, max_steps: int) -> np.ndarray:
    """Same linear-interpolation idea as dataset.py's
    _resample_to_fixed_steps, but the OUTPUT step count is derived from
    the segment's own duration at a fixed rate, not fixed outright — a
    trial twice as long gets roughly twice as many steps (capped at
    max_steps as a safety ceiling, see config_ctc.IMU_MAX_STEPS_CAP)."""
    order = np.argsort(t)
    t_sorted = t[order]
    values_sorted = values[order]
    duration = max(t_sorted[-1] - t_sorted[0], 1e-3)
    n_steps = max(2, min(int(round(duration * target_hz)), max_steps))
    t_query = np.linspace(t_sorted[0], t_sorted[-1], n_steps)
    out = np.empty((values.shape[1], n_steps), dtype=np.float32)
    for c in range(values.shape[1]):
        out[c] = np.interp(t_query, t_sorted, values_sorted[:, c])
    return out


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Centered moving-average smoothing along the time axis (values
    shape: (channels, T)) — see config_ctc.IMU_SMOOTH_WINDOW_SEC's
    docstring for why: camera-tracked fingertip accel/gyro/position has
    real, fast frame-to-frame jitter, much higher-frequency than any
    genuine letter-writing motion, so smoothing removes noise without
    blurring real motion features together. window <= 1 or too few
    samples to smooth meaningfully is a no-op (returns values unchanged)
    rather than raising — this can legitimately happen on a very short
    trial/segment. Edge-padded ('replicate' the first/last real value)
    so the output length always exactly matches the input, including at
    the very start/end where a centered window would otherwise need
    samples that don't exist."""
    if window <= 1 or values.shape[1] < 2:
        return values
    kernel = np.ones(window, dtype=np.float32) / window
    pad = window // 2
    smoothed = np.empty_like(values)
    for c in range(values.shape[0]):
        padded = np.pad(values[c], (pad, pad), mode='edge')
        smoothed[c] = np.convolve(padded, kernel, mode='valid')[:values.shape[1]]
    return smoothed


def load_imu_variable(trial_dir: Path, imu_source: str, finger: str,
                       t_start: float = None, t_end: float = None) -> torch.Tensor:
    """(6, T) accel+gyro sequence, resampled at a fixed RATE (config_ctc.
    IMU_RESAMPLE_HZ) rather than to a fixed STEP COUNT — see module
    docstring. Returns an all-zero (6, 2) placeholder (matching this
    function's minimum possible real output) if the source data can't
    support even a 2-point interpolation, mirroring dataset.py's
    all-zero-fallback behavior for unusable trials."""
    if imu_source == 'fingertip':
        df = pd.read_csv(trial_dir / config_ctc.IMU_FILENAMES['fingertip'])
        df = df[(df['finger'] == finger) & (df['detected'])]
        if t_start is not None:
            df = df[df['time_aligned'] >= t_start]
        if t_end is not None:
            df = df[df['time_aligned'] <= t_end]
        if len(df) < 2:
            return torch.zeros(len(config_ctc.IMU_CHANNELS), 2, dtype=torch.float32)
        t = df['time_aligned'].to_numpy()
        values = df[config_ctc.IMU_CHANNELS].to_numpy()
        imu = _resample_to_rate(t, values, config_ctc.IMU_RESAMPLE_HZ, config_ctc.IMU_MAX_STEPS_CAP)
    else:  # 'watch'
        df = pd.read_csv(trial_dir / config_ctc.IMU_FILENAMES['watch'])
        if t_start is not None:
            df = df[df['time_aligned'] >= t_start]
        if t_end is not None:
            df = df[df['time_aligned'] <= t_end]
        acc = df[df['sensor'] == 'acc']
        gyro = df[df['sensor'] == 'gyro']
        if len(acc) < 2 or len(gyro) < 2:
            return torch.zeros(len(config_ctc.IMU_CHANNELS), 2, dtype=torch.float32)
        acc_r = _resample_to_rate(acc['time_aligned'].to_numpy(), acc[['v1', 'v2', 'v3']].to_numpy(),
                                   config_ctc.IMU_RESAMPLE_HZ, config_ctc.IMU_MAX_STEPS_CAP)
        gyro_r = _resample_to_rate(gyro['time_aligned'].to_numpy(), gyro[['v1', 'v2', 'v3']].to_numpy(),
                                    config_ctc.IMU_RESAMPLE_HZ, config_ctc.IMU_MAX_STEPS_CAP)
        n = min(acc_r.shape[1], gyro_r.shape[1])   # acc/gyro can end up 1 step apart from
                                                     # independent rounding — trim to the shorter
        imu = np.concatenate([acc_r[:, :n], gyro_r[:, :n]], axis=0)

    imu = imu.astype(np.float32)
    smooth_window = max(1, int(config_ctc.IMU_SMOOTH_WINDOW_SEC * config_ctc.IMU_RESAMPLE_HZ))
    imu = _smooth(imu, smooth_window)   # reduce camera-tracking jitter before normalizing —
                                          # see _smooth()'s docstring
    mean = imu.mean(axis=1, keepdims=True)
    std = imu.std(axis=1, keepdims=True) + 1e-6
    imu = (imu - mean) / std
    return torch.from_numpy(imu)   # (6, T)


def discover_participant_dataset_dirs(participants_root: Path, participants: list = None,
                                       subfolder: str = 'dataset') -> list:
    """Finds every participant's data folder under participants_root —
    e.g. dataset/p1/dataset/, dataset/p3/dataset/, ... (participant name
    -> that participant's own <subfolder>/ inside it, matching the actual
    on-disk layout: dataset/<participant>/<subfolder>/<class>/trial_XXX/).

    Auto-discovers every "p*"-named folder under participants_root unless
    `participants` restricts it to a specific subset (e.g. ['p1', 'p3']).
    New participant folders added later (p4, p5, ...) are picked up
    automatically on the next run — no code change or extra CLI argument
    needed each time, which is the whole point: train_ctc.py/
    evaluate_word_ctc.py should never need editing just because another
    participant's data showed up.

    Returns [(participant_name, dataset_dir), ...], sorted by name. A
    participant folder that exists but has no <subfolder>/ inside it is
    skipped with a warning rather than silently dropped or crashing —
    could mean that participant's data collection is still in progress."""
    if participants:
        candidate_names = sorted(participants)
    else:
        if not participants_root.exists():
            return []
        candidate_names = sorted(p.name for p in participants_root.iterdir()
                                  if p.is_dir() and p.name.startswith('p'))

    dirs = []
    for name in candidate_names:
        d = participants_root / name / subfolder
        if d.exists():
            dirs.append((name, d))
        else:
            print(f'[WARN] participant "{name}" has no {subfolder}/ folder under '
                  f'{participants_root / name} — skipping')
    return dirs


def load_trajectory_variable(trial_dir: Path, finger: str,
                              t_start: float = None, t_end: float = None) -> "torch.Tensor | None":
    """(2, T) NORMALIZED fingertip (x, y) trajectory, resampled at the
    same rate as load_imu_variable — the real-world signal the model's
    "latent fingertip motion" bottleneck (see model_ctc.py's motion_head/
    traj_decoder) is trained to be reconstructible from via an auxiliary
    loss, the same role spatiotemporal mouth geometry plays in Watch Your
    Mouth: the network is made to route everything through a physically-
    grounded intermediate representation before decoding to text, instead
    of classifying directly off raw fused features.

    "Normalized" specifically means: NO absolute world coordinates.
    Per-trial mean-centered and std-scaled, so what's left is the SHAPE
    of the motion over time (relative to itself) — not where on the
    writing surface the participant happened to start that particular
    trial, which varies trial to trial and carries no information about
    which letter was written. This is exactly the property that makes
    the representation learnable as a general "how does writing move"
    concept rather than something tied to one physical setup.

    Read from fingertip_imu.csv's pos_x/pos_y columns — written
    regardless of which imu_source the PRIMARY input actually uses, since
    this is purely a training-time supervisory signal, never fed to the
    model directly (no risk of leaking position data into the
    classifier's actual input — the model only ever sees audio/IMU; this
    trajectory exists solely to shape what the motion_head bottleneck
    learns to represent). Z is dropped — 2D is enough for writing on a
    flat surface.

    Returns None (not a placeholder/zero tensor) when there's no usable
    position data for this trial (missing file, or fewer than 2 detected
    fingertip samples in the window — e.g. poor camera tracking) —
    callers should skip the auxiliary loss for that sample rather than
    train against a fabricated trajectory."""
    path = trial_dir / config_ctc.IMU_FILENAMES['fingertip']   # fingertip_imu.csv — always this file,
                                                                 # independent of the run's own --imu-source
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df[(df['finger'] == finger) & (df['detected'])]
    if t_start is not None:
        df = df[df['time_aligned'] >= t_start]
    if t_end is not None:
        df = df[df['time_aligned'] <= t_end]
    if len(df) < 2:
        return None
    t = df['time_aligned'].to_numpy()
    values = df[['pos_x', 'pos_y']].to_numpy()
    traj = _resample_to_rate(t, values, config_ctc.IMU_RESAMPLE_HZ, config_ctc.IMU_MAX_STEPS_CAP)
    smooth_window = max(1, int(config_ctc.IMU_SMOOTH_WINDOW_SEC * config_ctc.IMU_RESAMPLE_HZ))
    traj = _smooth(traj, smooth_window)   # a smoothed target is a fairer thing to ask the IMU
                                            # decoder to reconstruct than raw tracking jitter — see
                                            # _smooth()'s docstring
    mean = traj.mean(axis=1, keepdims=True)
    std = traj.std(axis=1, keepdims=True) + 1e-6
    traj = (traj - mean) / std   # absolute position removed — only relative shape remains
    return torch.from_numpy(traj.astype(np.float32))   # (2, T)


class LetterDatasetCTC(Dataset):
    """samples: [(trial_dir, label), ...] as returned by dataset.py's
    scan_dataset() — reused unmodified, only the loading below differs.
    label here is the trial's index into `classes` (0-25 for a-z); the
    CTC target sequence returned is always length 1 (just [label+1], +1
    to skip past index 0 which config_ctc.BLANK_IDX reserves)."""

    def __init__(self, samples: list, finger: str = config_ctc.DEFAULT_FINGER,
                 audio_source: str = config_ctc.DEFAULT_AUDIO_SOURCE,
                 imu_source: str = config_ctc.DEFAULT_IMU_SOURCE):
        if not samples:
            raise RuntimeError('LetterDatasetCTC got an empty sample list')
        self.samples = samples
        self.finger = finger
        self.audio_source = audio_source
        self.imu_source = imu_source

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        trial_dir, label = self.samples[idx]
        audio = load_audio_variable(trial_dir, self.audio_source)
        imu = load_imu_variable(trial_dir, self.imu_source, self.finger)
        traj = load_trajectory_variable(trial_dir, self.finger)   # None if unavailable — see its docstring
        # Surface mic spectrogram, ALWAYS from the 'surface' source regardless
        # of what self.audio_source actually is — the Mic decoder's
        # reconstruction target, mirroring exactly how the IMU decoder's
        # target is always fingertip_imu.csv regardless of self.imu_source
        # (see load_trajectory_variable). When self.audio_source is already
        # 'surface' this is trivially the same tensor as `audio` above (a
        # harmless, expected degenerate case — see train_ctc.py's
        # --spec-loss-weight docs).
        surface_audio = load_audio_variable(trial_dir, 'surface')
        target = torch.tensor([label + 1], dtype=torch.long)   # +1: skip past BLANK_IDX=0
        return audio, imu, target, traj, surface_audio


def collate_fn_ctc(batch):
    """Pads every sample's audio/IMU to the BATCH's own max length (never
    a fixed global length — see module docstring) and returns everything
    nn.CTCLoss expects: concatenated targets + per-sample target_lengths,
    plus per-sample INPUT lengths (pre-padding) so the loss/decoder never
    reads padding as real signal.

    audio: list of (1, N_MELS, T_i) -> (B, 1, N_MELS, T_max), zero-padded
    on the time axis. imu: list of (6, T_i) -> (B, 6, T_max), same idea.

    trajectory: list of (2, T_i)-or-None -> (B, 2, T_max) zero-padded,
    plus traj_lengths (0 for a sample whose trajectory was None) and
    has_traj (a bool mask) — see load_trajectory_variable()'s docstring
    for why some samples legitimately have no trajectory ground truth
    (poor fingertip tracking on that trial) rather than a fabricated one.
    Callers computing the auxiliary motion loss (see train_ctc.py) must
    mask by has_traj/traj_lengths, never assume every sample in a batch
    has a real trajectory target.

    surface_audio: list of (1, N_MELS, T_i) — the Mic decoder's
    reconstruction TARGET (always from the 'surface' mic source,
    regardless of what audio_source the model's actual INPUT uses — see
    LetterDatasetCTC.__getitem__) -> (B, 1, N_MELS, T_max), padded
    independently from the input `audio` above, since the two can have
    slightly different native frame counts (see train_ctc.py's
    compute_spec_loss, which interpolates to reconcile this)."""
    audios, imus, targets, trajs, surface_audios = zip(*batch)

    audio_lengths = torch.tensor([a.shape[2] for a in audios], dtype=torch.long)
    audio_max_t = int(audio_lengths.max())
    audio_padded = torch.stack([
        F.pad(a, (0, audio_max_t - a.shape[2])) for a in audios
    ])   # (B, 1, N_MELS, T_max)

    imu_lengths = torch.tensor([i.shape[1] for i in imus], dtype=torch.long)
    imu_max_t = int(imu_lengths.max())
    imu_padded = torch.stack([
        F.pad(i, (0, imu_max_t - i.shape[1])) for i in imus
    ])   # (B, 6, T_max)

    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    targets_concat = torch.cat(targets)   # CTCLoss wants targets flattened, not padded

    has_traj = torch.tensor([t is not None for t in trajs], dtype=torch.bool)
    traj_lengths = torch.tensor([(t.shape[1] if t is not None else 0) for t in trajs], dtype=torch.long)
    traj_max_t = int(traj_lengths.max()) if has_traj.any() else 1
    traj_padded = torch.zeros(len(trajs), 2, max(traj_max_t, 1), dtype=torch.float32)
    for i, t in enumerate(trajs):
        if t is not None:
            traj_padded[i, :, :t.shape[1]] = t

    surface_lengths = torch.tensor([s.shape[2] for s in surface_audios], dtype=torch.long)
    surface_max_t = int(surface_lengths.max())
    surface_padded = torch.stack([
        F.pad(s, (0, surface_max_t - s.shape[2])) for s in surface_audios
    ])   # (B, 1, N_MELS, T_max)

    return (audio_padded, audio_lengths, imu_padded, imu_lengths, targets_concat, target_lengths,
            traj_padded, traj_lengths, has_traj, surface_padded, surface_lengths)
