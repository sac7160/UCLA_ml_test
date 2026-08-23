"""
config_ctc.py
────────────────────────────────────────────────────────────────────────────
Constants for the CTC-based letter model — kept entirely separate from
config.py so nothing here can affect the original fixed-length pipeline
(train.py/test.py/model.py/dataset.py — all untouched).

Why a separate pipeline at all: the original one resamples every trial to
a FIXED number of steps (IMU_STEPS=64 regardless of duration, AUDIO_MAX_SEC
pad/truncate) — exactly what a CTC model can't use. CTC needs the time
axis to still reflect real relative timing (a trial twice as long should
produce roughly twice as many timesteps), and it needs to run on whole,
un-segmented word trials that are longer than any single letter trial —
touch_on/touch_off boundaries *inside* a word aren't reliable enough to
pre-segment on (see the chat this was designed in), so the model has to
find its own alignment between the continuous signal and the letter
sequence, which is exactly what CTC is for.

So this pipeline resamples proportionally to duration instead of to a
fixed step count, and pads to each batch's own max length at collate time
rather than to one global fixed length — see dataset_ctc.py.
"""

import config as base_config   # the original config.py — read-only reuse, so both pipelines
                                # stay consistent about what a "surface_mic.wav" or
                                # "fingertip_imu.csv" actually contains

AUDIO_TARGET_SR = base_config.AUDIO_TARGET_SR
N_MELS          = base_config.N_MELS
N_FFT           = base_config.N_FFT
HOP_LENGTH      = base_config.HOP_LENGTH
AUDIO_FILENAMES = base_config.AUDIO_FILENAMES
IMU_FILENAMES   = base_config.IMU_FILENAMES
IMU_CHANNELS    = base_config.IMU_CHANNELS
DEFAULT_FINGER        = base_config.DEFAULT_FINGER
DEFAULT_AUDIO_SOURCE  = base_config.DEFAULT_AUDIO_SOURCE
DEFAULT_IMU_SOURCE    = base_config.DEFAULT_IMU_SOURCE

# ─── Variable-length preprocessing ─────────────────────────────────────────────
IMU_RESAMPLE_HZ  = 40.0   # fixed RATE (not a fixed step count) to resample IMU to — a trial
                           # twice as long ends up with roughly twice as many IMU steps
AUDIO_MAX_SEC_CAP = 8.0    # hard ceiling on any single trial's audio length — generous enough
                           # for a whole multi-letter word, just a safety cap against one
                           # malformed/huge trial blowing up a batch's padding
IMU_MAX_STEPS_CAP = int(AUDIO_MAX_SEC_CAP * IMU_RESAMPLE_HZ)
IMU_SMOOTH_WINDOW_SEC = 0.075   # ~75ms centered moving-average smoothing applied to every resampled
                                 # IMU/trajectory signal (see dataset_ctc.py's _smooth()) — camera-tracked
                                 # fingertip accel/gyro/position has real, fast frame-to-frame jitter that's
                                 # much higher-frequency than any genuine letter-writing motion, so a short
                                 # window removes noise without blurring real motion features together. 0
                                 # disables smoothing entirely (raw resampled values, unmodified).

# ─── CTC label encoding ─────────────────────────────────────────────────────────
# CTC's blank class conventionally lives at index 0; every real letter
# class is offset by +1 from its position in `classes` (a-z -> indices
# 1-26). See dataset_ctc.py's encode_label() / model_ctc.py's head size
# (len(classes) + 1).
BLANK_IDX = 0
