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
from scipy.signal import medfilt, savgol_filter
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


def _robust_origin_normalize(values: np.ndarray, n_origin: int = 5) -> np.ndarray:
    """Subtracts a ROBUST LOCAL ORIGIN — the median of the first
    n_origin samples in the window — from every channel, WITHOUT
    dividing by standard deviation. This is the alternative to z-score
    normalization (mean-center + divide by std) used for the fingertip
    IMU TEACHER signal specifically (see load_imu_variable's
    robust_origin param): during a near-stationary window, z-score's std
    can be tiny, so dividing by it blows small camera-tracking jitter
    (MediaPipe noise) up to look like large, meaningful movement — the
    opposite of what a clean training target should show. Subtracting a
    robust origin instead removes the arbitrary absolute starting value
    (which trial to trial has no meaning — where on the surface the
    participant happened to start) while preserving REAL magnitude: a
    genuinely stationary window stays near zero, and genuine movement
    stays distinguishable by its actual size, not rescaled relative to
    that window's own (possibly tiny, possibly huge) variance.

    The median (not just the first sample) of the first few points is
    used specifically because a single first sample could itself be an
    outlier/jitter spike — the median of several is more robust to
    exactly the kind of one-off noise this is trying to avoid amplifying
    in the first place."""
    n = min(n_origin, values.shape[1])
    origin = np.median(values[:, :n], axis=1, keepdims=True)
    return values - origin


def _denoise_fingertip_signal(values: np.ndarray) -> np.ndarray:
    """Two-stage denoising for any fingertip IMU signal — applied
    whenever imu_source=='fingertip' in load_imu_variable, regardless of
    whether that call is loading the model's primary INPUT (e.g. a
    fingertip-IMU-only ablation run) or the IMU decoder's TEACHER
    target: MediaPipe's webcam-based tracking jitter is a property of
    the camera-tracked source itself, not of how the loaded signal
    happens to be used afterward, so both roles benefit equally. On top
    of the existing _smooth() moving average, since MediaPipe's
    webcam-based hand tracking has TWO distinct kinds of noise that a
    single moving-average filter doesn't handle equally well:

    1) Occasional single-frame SPIKES (occlusion, a momentary bad
       detection jumping to a wildly wrong value) — a moving average
       just blends a spike into its neighbors, still distorting them;
       a median filter instead REJECTS the spike outright (a median is
       far more robust to one extreme outlier in the window than a
       mean is), so it's applied FIRST, before any smoothing that
       averages values together.

    2) Continuous low-level JITTER (small frame-to-frame tracking
       noise even when nothing is occluded) — for this, a
       Savitzky-Golay filter (fits a local polynomial per window,
       evaluated at the window's center) smooths while preserving the
       actual SHAPE of real motion — its peaks, slopes, curvature —
       far better than a plain moving average does, which tends to
       flatten and lag behind genuine fast movements.

    Runs BEFORE _smooth() and _robust_origin_normalize() in
    load_imu_variable — median filter needs the least-processed signal
    to correctly identify true outliers (smoothing first would already
    have blended a spike into its neighbors, making it harder to
    detect as an outlier)."""
    T = values.shape[1]
    if T < 5:
        return values   # too short for either filter to do anything meaningful — matches
                          # _smooth()'s own short-trial no-op behavior

    # Stage 1: median filter — kernel size must be odd; 5 samples (~125ms
    # at IMU_RESAMPLE_HZ=40Hz) is short enough to only catch brief
    # spikes, not blur out real, slightly-longer motion features.
    median_kernel = min(5, T if T % 2 == 1 else T - 1)
    if median_kernel >= 3:
        despiked = np.stack([medfilt(values[c], kernel_size=median_kernel) for c in range(values.shape[0])])
    else:
        despiked = values

    # Stage 2: Savitzky-Golay — window must be odd and > polyorder;
    # polyorder=2 (fits a local parabola) is enough to track real
    # acceleration/velocity curvature without overfitting to jitter
    # inside a short window.
    savgol_window = min(9, T if T % 2 == 1 else T - 1)
    if savgol_window > 3:
        smoothed = savgol_filter(despiked, window_length=savgol_window, polyorder=2, axis=1)
    else:
        smoothed = despiked
    return smoothed.astype(np.float32)


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
                       t_start: float = None, t_end: float = None,
                       robust_origin: bool = False) -> torch.Tensor:
    """(6, T) accel+gyro sequence, resampled at a fixed RATE (config_ctc.
    IMU_RESAMPLE_HZ) rather than to a fixed STEP COUNT — see module
    docstring. Returns an all-zero (6, 2) placeholder (matching this
    function's minimum possible real output) if the source data can't
    support even a 2-point interpolation, mirroring dataset.py's
    all-zero-fallback behavior for unusable trials.

    robust_origin: which normalization to apply after smoothing —
    False (default, used for the model's actual INPUT regardless of
    imu_source) keeps the existing z-score (mean-center + divide by std)
    normalization, which is fine for an input signal a downstream
    encoder will learn to work with either way. True (used specifically
    when this same function loads the fingertip-IMU TEACHER signal — see
    dataset_ctc.LetterDatasetCTC/dataset_ctc_realtext.RealTextDatasetCTC's
    imu_target) switches to _robust_origin_normalize instead: z-score can
    amplify small camera-tracking jitter during a near-stationary window
    (tiny std there means dividing by it blows noise up to look like
    real movement) — exactly the wrong behavior for something a decoder
    is being trained to faithfully reconstruct. See
    _robust_origin_normalize's own docstring for the full reasoning.

    NOTE: the median-filter + Savitzky-Golay denoising (see
    _denoise_fingertip_signal) is applied whenever imu_source=='fingertip'
    — regardless of robust_origin, i.e. regardless of whether this call
    is loading the teacher OR the model's actual primary input (e.g. a
    fingertip-IMU-only ablation run, --imu-source fingertip). This is
    deliberate: MediaPipe's webcam-based tracking jitter is a property of
    the SOURCE (camera-tracked fingertip data), not of how the loaded
    signal happens to be used — a fingertip-only INPUT model benefits
    from the same denoising a teacher does, for the same underlying
    reason. Only the NORMALIZATION scheme (z-score vs robust-origin)
    still depends on robust_origin/role, since that choice is about how
    directly the signal gets compared in a loss (see above), which
    denoising has no bearing on. Watch IMU is a real physical sensor
    with different noise characteristics (thermal/electrical, not
    camera-tracking jitter) and is never run through this denoising."""
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
    if imu_source == 'fingertip':
        # Applies regardless of role (teacher OR primary model input) —
        # MediaPipe tracking jitter is a property of the SOURCE, not the
        # use-case. Median filter (reject single-frame occlusion/
        # detection spikes) then Savitzky-Golay (smooth while preserving
        # real motion shape), BEFORE the plain moving average below,
        # since the median filter needs the least-processed signal to
        # correctly tell a real spike apart from real motion — see
        # _denoise_fingertip_signal's own docstring. Watch IMU never
        # reaches this branch (see the if/else above) — a real physical
        # sensor's noise characteristics are different, so it isn't run
        # through camera-tracking-specific denoising.
        imu = _denoise_fingertip_signal(imu)
    smooth_window = max(1, int(config_ctc.IMU_SMOOTH_WINDOW_SEC * config_ctc.IMU_RESAMPLE_HZ))
    imu = _smooth(imu, smooth_window)   # reduce camera-tracking jitter before normalizing —
                                          # see _smooth()'s docstring
    if robust_origin:
        imu = _robust_origin_normalize(imu)
    else:
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
        # Fingertip IMU, ALWAYS from the 'fingertip' source regardless of
        # what self.imu_source actually is — the IMU decoder's
        # reconstruction target, mirroring exactly how the Mic decoder's
        # target is always the 'surface' mic source regardless of
        # self.audio_source (see surface_audio below). When self.imu_source
        # is already 'fingertip' this is trivially the same tensor as
        # `imu` above (a harmless, expected degenerate case — see
        # train_ctc.py's --motion-loss-weight docs). Never None — unlike
        # the old trajectory target, load_imu_variable always returns a
        # real (possibly all-zero-fallback) tensor, so there's no
        # has-target masking to carry through collate_fn_ctc/the loss.
        imu_target = load_imu_variable(trial_dir, 'fingertip', self.finger, robust_origin=True)
        # Surface mic spectrogram, ALWAYS from the 'surface' source regardless
        # of what self.audio_source actually is — the Mic decoder's
        # reconstruction target, mirroring exactly how the IMU decoder's
        # target is always fingertip IMU regardless of self.imu_source
        # above. When self.audio_source is already 'surface' this is
        # trivially the same tensor as `audio` above (a harmless, expected
        # degenerate case — see train_ctc.py's --spec-loss-weight docs).
        surface_audio = load_audio_variable(trial_dir, 'surface')
        target = torch.tensor([label + 1], dtype=torch.long)   # +1: skip past BLANK_IDX=0
        return audio, imu, target, imu_target, surface_audio


def collate_fn_ctc(batch):
    """Pads every sample's audio/IMU to the BATCH's own max length (never
    a fixed global length — see module docstring) and returns everything
    nn.CTCLoss expects: concatenated targets + per-sample target_lengths,
    plus per-sample INPUT lengths (pre-padding) so the loss/decoder never
    reads padding as real signal.

    audio: list of (1, N_MELS, T_i) -> (B, 1, N_MELS, T_max), zero-padded
    on the time axis. imu: list of (6, T_i) -> (B, 6, T_max), same idea.

    imu_target: list of (6, T_i) — the IMU decoder's reconstruction
    TARGET (always from the 'fingertip' source, regardless of what
    imu_source the model's actual INPUT uses — see
    LetterDatasetCTC.__getitem__) -> (B, 6, T_max), padded independently
    from the input `imu` above, since the two can have slightly different
    native frame counts (see train_ctc.py's compute_imu_recon_loss, which
    interpolates to reconcile this). Unlike the old trajectory target,
    load_imu_variable never returns None (an unusable trial falls back to
    an all-zero placeholder instead), so there's no has-target mask to
    carry through here.

    surface_audio: list of (1, N_MELS, T_i) — the Mic decoder's
    reconstruction TARGET (always from the 'surface' mic source,
    regardless of what audio_source the model's actual INPUT uses — see
    LetterDatasetCTC.__getitem__) -> (B, 1, N_MELS, T_max), padded
    independently from the input `audio` above, since the two can have
    slightly different native frame counts (see train_ctc.py's
    compute_spec_loss, which interpolates to reconcile this)."""
    audios, imus, targets, imu_targets, surface_audios = zip(*batch)

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

    imu_target_lengths = torch.tensor([t.shape[1] for t in imu_targets], dtype=torch.long)
    imu_target_max_t = int(imu_target_lengths.max())
    imu_target_padded = torch.stack([
        F.pad(t, (0, imu_target_max_t - t.shape[1])) for t in imu_targets
    ])   # (B, 6, T_max)

    surface_lengths = torch.tensor([s.shape[2] for s in surface_audios], dtype=torch.long)
    surface_max_t = int(surface_lengths.max())
    surface_padded = torch.stack([
        F.pad(s, (0, surface_max_t - s.shape[2])) for s in surface_audios
    ])   # (B, 1, N_MELS, T_max)

    return (audio_padded, audio_lengths, imu_padded, imu_lengths, targets_concat, target_lengths,
            imu_target_padded, imu_target_lengths, surface_padded, surface_lengths)