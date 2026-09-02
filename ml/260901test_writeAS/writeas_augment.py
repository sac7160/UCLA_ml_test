"""
writeas_augment.py
────────────────────────────────────────────────────────────────────────────
The four data-augmentation mechanisms WriteAS (Zhang et al., IMWUT 2021)
describes in Section 4.1/5.1.4 — see model_writeas.py's own docstring for
the rest of the replication. All four operate on (6, T) IMU tensors
(accel_x/y/z, gyro_x/y/z — this project's own channel convention, see
config_ctc.IMU_CHANNELS) and are applied ON-THE-FLY per training sample
(not as a fixed offline-generated set) — matching how this project's own
existing synthetic-concatenation augmentation
(dataset_ctc_concat.ConcatLetterDatasetCTC) already works, rather than
the paper's own literal "generate a fixed 5x-larger dataset up front"
framing, which doesn't fit this project's Dataset/DataLoader pattern as
cleanly.

1. time_warp — Section 4.1: "given the motion signal x(t), we expand or
   contract it in the time axis by a factor alpha... thereby generating
   a new profile x(alpha*t)". alpha in {0.5, 0.75, 1.25, 1.5} (paper's
   own stated candidate set).
2. segment_warp — Section 4.1's "another warping mechanism": randomly
   select n=2 segments of length W=0.5s and warp them along the time
   axis (same alpha candidates as time_warp, applied LOCALLY to just
   those segments rather than the whole signal).
3. rotation_injection — Section 4.1, Eq. 4: R(theta) rotation about the
   arm axis (x-axis), theta in [-30, 30] degrees at 5-degree steps (the
   paper's own stated range/step, "12 new profiles"). Applied to BOTH
   the accel and gyro 3-axis groups independently, since both are
   3-axis vector signals whose axes rotate together with the watch.
4. word_concat — Section 4.1's "concatenate two samples... from the same
   person" — implemented as a Dataset class (WriteASConcatDataset in
   train_writeas.py, not here) rather than a plain function, since it
   needs to pick a SECOND sample (from the same participant) rather than
   transforming one sample in isolation the way the other three do.
"""

import math
import random

import torch
import torch.nn.functional as F

TIME_WARP_ALPHAS = (0.5, 0.75, 1.25, 1.5)   # paper's own stated candidate set (Section 4.1)
ROTATION_THETA_RANGE_DEG = 30   # paper's own stated range: [-30, 30]
ROTATION_THETA_STEP_DEG = 5     # paper's own stated step


def time_warp(imu: torch.Tensor, alpha: float = None) -> torch.Tensor:
    """(6, T) -> (6, T') where T' = round(T / alpha) — resamples the
    WHOLE signal to a new duration, matching x(alpha*t): alpha>1
    compresses (faster/shorter), alpha<1 stretches (slower/longer). Picks
    a random alpha from TIME_WARP_ALPHAS if none given."""
    alpha = alpha if alpha is not None else random.choice(TIME_WARP_ALPHAS)
    T = imu.shape[1]
    new_T = max(2, int(round(T / alpha)))
    return F.interpolate(imu.unsqueeze(0), size=new_T, mode='linear', align_corners=False).squeeze(0)


def segment_warp(imu: torch.Tensor, sample_rate: float, n_segments: int = 2,
                  segment_sec: float = 0.5, alphas=TIME_WARP_ALPHAS) -> torch.Tensor:
    """(6, T) -> (6, T'') — picks n_segments non-overlapping segments of
    length segment_sec (paper's own W=0.5s), each independently
    time-warped by its own random alpha and spliced back into the
    sequence (Section 4.1's second warping mechanism). A no-op (returns
    imu unchanged) when the signal is too short to carve out even one
    such segment."""
    T = imu.shape[1]
    seg_len = max(1, int(round(segment_sec * sample_rate)))
    if T <= seg_len:
        return imu
    n_possible = max(1, T - seg_len)
    n_pick = min(n_segments, n_possible)
    starts = sorted(random.sample(range(n_possible), n_pick))
    # Drop any start that would overlap a PREVIOUSLY picked segment (rare
    # with real trial lengths, but keeps this correct rather than
    # producing a mangled splice on a short/heavily-picked signal).
    kept_starts = []
    cursor = -1
    for s in starts:
        if s >= cursor:
            kept_starts.append(s)
            cursor = s + seg_len

    pieces = []
    cursor = 0
    for start in kept_starts:
        if start > cursor:
            pieces.append(imu[:, cursor:start])
        seg = imu[:, start:start + seg_len]
        alpha = random.choice(alphas)
        new_len = max(1, int(round(seg_len / alpha)))
        seg_warped = F.interpolate(seg.unsqueeze(0), size=new_len, mode='linear',
                                    align_corners=False).squeeze(0)
        pieces.append(seg_warped)
        cursor = start + seg_len
    if cursor < T:
        pieces.append(imu[:, cursor:])
    return torch.cat(pieces, dim=1)


def rotation_injection(imu: torch.Tensor, theta_deg: float = None) -> torch.Tensor:
    """(6, T) -> (6, T), same shape — rotates BOTH the accel (channels
    0:3) and gyro (channels 3:6) 3-axis groups by R(theta) about the arm
    axis (x-axis), matching the paper's own Eq. 4:
        R(theta) = [[1,0,0],[0,cos(theta),-sin(theta)],[0,sin(theta),cos(theta)]]
    Picks a random theta from the paper's own stated
    [-30, 30] degree range at 5-degree steps if none given (excluding 0,
    which is just the unrotated original — "12 new profiles" per the
    paper's own count)."""
    if theta_deg is None:
        choices = [t for t in range(-ROTATION_THETA_RANGE_DEG, ROTATION_THETA_RANGE_DEG + 1,
                                     ROTATION_THETA_STEP_DEG) if t != 0]
        theta_deg = random.choice(choices)
    theta = math.radians(theta_deg)
    c, s = math.cos(theta), math.sin(theta)
    R = torch.tensor([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=imu.dtype)
    accel = imu[0:3, :]   # (3, T)
    gyro = imu[3:6, :]    # (3, T)
    accel_rot = R @ accel
    gyro_rot = R @ gyro
    return torch.cat([accel_rot, gyro_rot], dim=0)


def apply_random_augmentation(imu: torch.Tensor, sample_rate: float) -> torch.Tensor:
    """Applies exactly ONE of {no-op, time_warp, segment_warp,
    rotation_injection}, chosen uniformly at random, to one training
    sample — called once per __getitem__ in WriteASWordDataset when
    --augment is set (see train_writeas.py). Over many epochs, this has
    the same effect as the paper's own "expand the dataset Nx" framing
    (Section 4.1/5.1.4) without needing to materialize a fixed larger
    dataset up front — see this module's own top-level docstring."""
    choice = random.choice(['none', 'time_warp', 'segment_warp', 'rotation'])
    if choice == 'none':
        return imu
    if choice == 'time_warp':
        return time_warp(imu)
    if choice == 'segment_warp':
        return segment_warp(imu, sample_rate)
    return rotation_injection(imu)
