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

import os

import config as base_config   # the original config.py — read-only reuse, so both pipelines
                                # stay consistent about what a "surface_mic.wav" or
                                # "fingertip_imu.csv" actually contains

# Deliberately NOT inherited from base_config — the collector's own
# AUDIO_TARGET_SR was set for a different purpose (real-time collection),
# and this project's earlier default (16kHz) was based on general speech-
# recognition convention (most spoken-word energy sits under 8kHz). That
# convention doesn't directly apply here: friction/writing sounds are a
# physically different kind of signal than speech, and prior work
# measuring exactly this (SoundScroll, ISWC 2024 — wrist-worn mic
# detecting finger-surface friction sound) found usable signal up to
# 30kHz+, with accuracy dropping sharply once frequencies above 30kHz are
# cut off. 48kHz keeps the FULL originally-captured audio (both
# watch_audio.wav and surface_mic.wav were recorded at 48kHz — see the
# chat this was changed in) with NO resampling-induced information loss
# at all, at the cost of ~3x the mel-spectrogram size/compute versus the
# earlier 16kHz default. If audio_source's own native sample rate is
# already 48000 (checked in load_audio_variable via torchaudio.resample,
# which no-ops when sr already matches), this setting means "just use
# the recording exactly as captured" — no resampling happens either way.
AUDIO_TARGET_SR = 48000
N_MELS          = base_config.N_MELS
# Read from an environment variable (falling back to base_config's own
# value when unset) specifically so sweep_audio_features.py can try
# several window/hop settings without editing this file — each candidate
# launches train_ctc.py as a FRESH subprocess with CTC_N_FFT/
# CTC_HOP_LENGTH set, since dataset_ctc.py bakes these into a module-
# level MelSpectrogram transform (_MEL) the first time it's imported;
# trying several values within one long-running process wouldn't change
# anything after the first import. See sweep_audio_features.py's own
# docstring.
N_FFT           = int(os.environ.get('CTC_N_FFT', base_config.N_FFT))
HOP_LENGTH      = int(os.environ.get('CTC_HOP_LENGTH', base_config.HOP_LENGTH))
AUDIO_FILENAMES = base_config.AUDIO_FILENAMES
IMU_FILENAMES   = base_config.IMU_FILENAMES
IMU_CHANNELS    = base_config.IMU_CHANNELS
DEFAULT_FINGER        = base_config.DEFAULT_FINGER
DEFAULT_AUDIO_SOURCE  = base_config.DEFAULT_AUDIO_SOURCE
DEFAULT_IMU_SOURCE    = base_config.DEFAULT_IMU_SOURCE

TRAJECTORY_FILENAME = 'trajectory_smooth_120hz.csv'   # offline-retracked fingertip (x, y) position
                                                          # (see extract_fingertip_trajectory.py) — NOT
                                                          # config.py's own live-collection trajectory.csv,
                                                          # this is a separate, higher-quality re-track
                                                          # done afterward directly off the raw video. Its
                                                          # own timestamps are already evenly spaced (no
                                                          # watch-style batching to fix — verified against
                                                          # real files in the chat this was added in), but
                                                          # NOT actually 120Hz despite the filename — the
                                                          # real per-file rate is derived from its own
                                                          # timestamps at load time, never assumed.
TRAJECTORY_PIXEL_SCALE = 100.0   # divides trajectory (x_px, y_px) AFTER robust-origin subtraction —
                                    # real trajectory files verified in this project have pixel
                                    # magnitudes in the tens-to-hundreds range, while fingertip IMU's
                                    # own accel/gyro values are typically well under 1 — at the SAME
                                    # --motion-loss-weight, an unscaled trajectory target's MSE loss
                                    # can be many orders of magnitude larger than fingertip_imu's own,
                                    # letting the motion loss completely dominate/destabilize CTC
                                    # training rather than gently guiding it (see the chat this was
                                    # added in — "no improvement" turned out to likely be this). A
                                    # FIXED constant, not a per-window statistic (e.g. std or MAD),
                                    # is used deliberately — the same "near-stationary window blows up
                                    # jitter" pitfall _robust_origin_normalize's own docstring
                                    # documents for fingertip IMU applies equally here if the divisor
                                    # were computed per-window instead of fixed.

# ─── Variable-length preprocessing ─────────────────────────────────────────────
IMU_RESAMPLE_HZ  = 100.0   # fixed RATE (not a fixed step count) to resample IMU to — a trial
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