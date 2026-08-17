"""
correlate_drop_with_accuracy.py
────────────────────────────────────────────────────────────────────────────
Directly tests whether fingertip_imu.csv's frame-drop rate actually
predicts misclassification — using an already-trained fingertip-IMU
checkpoint and its held-out test split. No new data collection or
retraining needed.

For each test trial: compute its frame-drop rate (gaps in unique camera
timestamps > 1.5x the trial's own median interval — the same detection
method used throughout this investigation) and the model's prediction
(correct/incorrect). If drop rate predicts errors, that's real evidence
frame drops are (at least partly) responsible for fingertip IMU's lower
accuracy vs. watch IMU; if there's no relationship, drops probably aren't
the (main) explanation, and the camera-fix investigation should look
elsewhere (e.g. position-estimation precision — see fingertip_position
in dataset.py/config.py) rather than at hardware like Leap Motion for
THIS specific reason (Leap Motion might still be worth it for other
reasons, but this test won't be one of them if the correlation comes back
weak).

Usage:
    python correlate_drop_with_accuracy.py \\
        --checkpoint checkpoints/letter_fingertip/best_model.pt \\
        --splits checkpoints/letter_fingertip/splits.json
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats as scipy_stats

import config
from dataset import DigitStrokeDataset, load_splits
from model import build_model, forward_model


def trial_drop_rate(trial_dir: Path) -> float | None:
    """Fraction of frame-to-frame intervals in fingertip_imu.csv that are
    over 1.5x the trial's own median interval. Returns None if the file
    is missing or too short to judge."""
    path = trial_dir / config.IMU_FILENAMES['fingertip']
    if not path.exists():
        return None
    df = pd.read_csv(path)
    t = df['time_aligned'].drop_duplicates().sort_values().to_numpy()
    if len(t) < 4:
        return None
    diffs = np.diff(t)
    median_interval = np.median(diffs)
    if median_interval <= 0:
        return None
    skips = diffs[diffs > 1.5 * median_interval]
    return len(skips) / len(diffs)


def main():
    parser = argparse.ArgumentParser(description='Correlate fingertip_imu.csv frame-drop rate with '
                                                  'model prediction correctness on the held-out test set')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--splits', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, default=Path('drop_correlation'))
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    splits, classes, finger, audio_source, imu_source, seed = load_splits(args.splits)
    if imu_source not in ('fingertip', 'fingertip_position'):
        print(f'[WARN] this checkpoint used imu_source="{imu_source}" — drop-rate analysis is only '
              f'meaningful for a fingertip-camera-based IMU source; results below may not mean much')

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    modality = ckpt.get('modality', 'fusion')
    model = build_model(modality, n_classes=len(classes)).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    test_ds = DigitStrokeDataset(splits['test'], finger=finger, audio_source=audio_source, imu_source=imu_source)

    rows = []
    with torch.no_grad():
        for i, (trial_dir, label) in enumerate(splits['test']):
            drop_rate = trial_drop_rate(trial_dir)
            if drop_rate is None:
                continue
            audio, imu, lbl = test_ds[i]
            audio_b, imu_b = audio.unsqueeze(0).to(device), imu.unsqueeze(0).to(device)
            logits = forward_model(model, modality, audio_b, imu_b)
            pred = logits.argmax(dim=1).item()
            rows.append({'trial': str(trial_dir), 'drop_rate': drop_rate,
                         'label': lbl, 'pred': pred, 'correct': pred == lbl})

    if not rows:
        raise RuntimeError('no test trials had a usable fingertip_imu.csv to compute drop rate from')

    df = pd.DataFrame(rows)
    print(f'[DATA] {len(df)} test trials with usable drop-rate info '
          f'(mean drop rate: {df["drop_rate"].mean() * 100:.1f}%, '
          f'overall test accuracy: {df["correct"].mean():.3f})')

    # point-biserial correlation: drop_rate (continuous) vs correct (binary) —
    # the standard correlation for a continuous variable against a 0/1 outcome
    corr, pval = scipy_stats.pointbiserialr(df['correct'].astype(int), df['drop_rate'])
    print(f'\npoint-biserial correlation(correct, drop_rate) = {corr:.3f}  (p={pval:.4f})')
    print('(negative and significant = higher drop rate really does go with more errors, '
          'as the "drops explain the accuracy gap" hypothesis predicts)')

    # bucket comparison: accuracy on the half of trials with the lowest vs.
    # highest drop rate — simpler to read at a glance than the raw correlation
    median_drop = df['drop_rate'].median()
    low = df[df['drop_rate'] <= median_drop]
    high = df[df['drop_rate'] > median_drop]
    acc_low = low['correct'].mean()
    acc_high = high['correct'].mean()
    print(f'\naccuracy on low-drop-rate half  (<= median {median_drop:.1%}): {acc_low:.3f}  (n={len(low)})')
    print(f'accuracy on high-drop-rate half (>  median {median_drop:.1%}): {acc_high:.3f}  (n={len(high)})')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    rng = np.random.default_rng(0)
    colors = ['#2ca02c' if c else '#d62728' for c in df['correct']]
    jitter = rng.uniform(-0.05, 0.05, len(df))
    ax1.scatter(df['drop_rate'] * 100, df['correct'].astype(int) + jitter, c=colors, alpha=0.6)
    ax1.set_xlabel('frame drop rate (%)'); ax1.set_ylabel('correct (1) / incorrect (0), jittered')
    ax1.set_title(f'Per-trial: drop rate vs. correctness\ncorr={corr:.3f}, p={pval:.4f}')
    ax1.grid(alpha=0.3)

    ax2.bar(['low drop-rate\n(<= median)', 'high drop-rate\n(> median)'], [acc_low, acc_high],
            color=['#1f77b4', '#d62728'])
    ax2.set_ylabel('accuracy'); ax2.set_ylim(0, 1.05)
    ax2.set_title('Accuracy: low vs. high drop-rate trials')
    ax2.grid(axis='y', alpha=0.3)

    fig.suptitle('Does fingertip camera frame-drop rate predict misclassification?', fontweight='bold')
    fig.tight_layout()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_dir / 'drop_vs_accuracy.png', dpi=150)
    df.to_csv(args.out_dir / 'per_trial_drop_and_correctness.csv', index=False)
    plt.close(fig)
    print(f'\n[PLOT] {args.out_dir}/drop_vs_accuracy.png')
    print(f'[DATA] {args.out_dir}/per_trial_drop_and_correctness.csv')


if __name__ == '__main__':
    main()