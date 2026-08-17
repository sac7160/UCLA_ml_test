"""
digit_classifier/config.py
────────────────────────────────────────────────────────────────────────────
Constants shared by dataset.py, model.py, train.py, and test.py. Change
these here rather than in multiple places — in particular, if AUDIO_* or
IMU_* change, any checkpoint trained under the old values won't load
correctly against a model built under the new ones (shapes won't match).
"""

# ─── Audio input (see dataset.py's DigitStrokeDataset._load_audio) ───────────
AUDIO_TARGET_SR = 16000
AUDIO_MAX_SEC   = 2.5
N_MELS          = 64
N_FFT           = 1024
HOP_LENGTH      = 256

# Which wav file counts as "the audio" for a trial — chosen via --audio-source.
AUDIO_FILENAMES = {
    'surface': 'surface_mic.wav',
    'watch':   'watch_audio.wav',
}

# ─── IMU input (see dataset.py's load_imu()) ─────────────────────────────────
# All sources end up as the same shape (6 channels x IMU_STEPS) — they just
# start from very differently-shaped CSVs, or different columns of the same
# one. See dataset.py's _load_imu_fingertip() / _load_imu_watch() /
# _load_imu_fingertip_pos_vel().
IMU_CHANNELS = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
IMU_STEPS    = 64
IMU_FILENAMES = {
    'fingertip':          'fingertip_imu.csv',
    'watch':               'imu.csv',
    'fingertip_position':  'fingertip_imu.csv',   # same file as 'fingertip' — pos_x/y/z + a single
                                                    # finite-difference velocity, instead of the
                                                    # doubly-differentiated accel/gyro columns. See
                                                    # dataset.py's _load_imu_fingertip_pos_vel().
}

# ─── Defaults (all overridable via CLI args in train.py / test.py) ───────────
DEFAULT_CLASSES = [f'digits_{d}' for d in range(10)]   # digits_0 .. digits_9
DEFAULT_FINGER  = 'index'          # only used when --imu-source fingertip
DEFAULT_AUDIO_SOURCE = 'surface'   # 'surface' (surface_mic.wav) or 'watch' (watch_audio.wav)
DEFAULT_IMU_SOURCE   = 'fingertip'   # 'fingertip' (fingertip_imu.csv) or 'watch' (imu.csv)