"""
digit_classifier/test.py
────────────────────────────────────────────────────────────────────────────
Two modes:

  --splits       (original use) evaluates the held-out test split saved by
                  train.py. Run this once, after training/model-selection is
                  completely done — running it repeatedly to "check
                  progress" and adjusting anything in response defeats the
                  point of holding a test set out; use val accuracy during
                  training for that instead.

  --dataset-root evaluates on EVERY trial under a different dataset root
                  instead — e.g. testing a trained model against a new
                  person's data, to check how well it generalizes to
                  someone whose writing it never saw during training. Uses
                  whichever classes/audio_source/imu_source/finger the
                  checkpoint itself was trained with (read straight from
                  the checkpoint), so there's no risk of accidentally
                  evaluating with mismatched settings.

Usage:
    python test.py --checkpoint checkpoints/digit_pilot/best_model.pt --splits checkpoints/digit_pilot/splits.json
    python test.py --checkpoint checkpoints/digit_pilot/best_model.pt --dataset-root ../other_person_dataset
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import DigitStrokeDataset, load_splits, scan_dataset
from model import build_model, forward_model


def evaluate(model, modality, loader, device, n_classes):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    with torch.no_grad():
        for audio, imu, label in loader:
            audio, imu, label = audio.to(device), imu.to(device), label.to(device)
            logits = forward_model(model, modality, audio, imu)
            loss = F.cross_entropy(logits, label)
            total_loss += loss.item() * label.size(0)
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_labels.extend(label.cpu().tolist())

    n = len(all_labels)
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / n

    confusion = [[0] * n_classes for _ in range(n_classes)]
    for p, l in zip(all_preds, all_labels):
        confusion[l][p] += 1

    return total_loss / n, acc, confusion


def print_report(confusion: list, classes: list):
    n = len(classes)
    short = [c[:10] for c in classes]

    print('\nConfusion matrix (rows = true label, cols = predicted):')
    print(' ' * 12 + ''.join(f'{c:>12}' for c in short))
    for i, row in enumerate(confusion):
        print(f'{short[i]:>12}' + ''.join(f'{v:>12d}' for v in row))

    print('\nPer-class precision / recall / F1:')
    for i, cls in enumerate(classes):
        tp = confusion[i][i]
        fp = sum(confusion[r][i] for r in range(n)) - tp
        fn = sum(confusion[i]) - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        support = sum(confusion[i])
        print(f'  {cls:12s} precision={precision:.3f}  recall={recall:.3f}  f1={f1:.3f}  (n={support})')


def plot_confusion_matrix(confusion: list, classes: list, test_acc: float, modality: str, save_path: Path):
    cm = np.array(confusion)
    # Row-normalized (recall per true class) for the color scale — raw counts
    # are still what's printed in each cell, since with imbalanced classes a
    # raw-count heatmap makes the majority class the only thing visible.
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(1.6 * len(classes) + 2, 1.4 * len(classes) + 2))
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax, label='fraction of true class')

    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    ax.set_xlabel('predicted'); ax.set_ylabel('true')
    ax.set_title(f'Test confusion matrix ({modality}) — acc={test_acc:.3f}, n={int(cm.sum())}')

    # Text color flips to white on dark cells so counts stay readable across
    # the whole colormap range, not just on the light end.
    for i in range(len(classes)):
        for j in range(len(classes)):
            color = 'white' if cm_norm[i, j] > 0.6 else 'black'
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', color=color, fontsize=11)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)



def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained checkpoint')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--splits', type=Path, default=None,
                         help='evaluate the held-out test split saved by train.py')
    parser.add_argument('--dataset-root', type=Path, default=None,
                         help='evaluate on ALL trials under a different dataset root instead — e.g. a '
                              'new person\'s data. Mutually exclusive with --splits.')
    parser.add_argument('--classes', nargs='+', default=None,
                         help='override the checkpoint\'s classes list — only used with --dataset-root, '
                              'e.g. if the new person only recorded some of the digits')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--plot-out', type=Path, default=None,
                         help='where to save the confusion matrix plot — defaults to '
                              'confusion_matrix.png next to --checkpoint')
    args = parser.parse_args()

    if (args.splits is None) == (args.dataset_root is None):
        parser.error('pass exactly one of --splits (the original held-out test split) or '
                      '--dataset-root (evaluate on a different dataset entirely)')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    modality = ckpt.get('modality', 'fusion')   # older checkpoints (pre-modality) were always fusion

    if args.dataset_root is not None:
        # Resolved to an absolute path before anything else — a relative
        # --dataset-root is resolved against wherever the shell's cwd
        # happens to be when you run this, which silently differs
        # depending on whether you're sitting in run/ or run/ml/ — printing
        # the absolute path makes it immediately obvious which directory
        # actually got scanned, instead of a bare "no folder" warning that
        # looks identical whether the path was merely wrong or the data
        # genuinely isn't there.
        args.dataset_root = args.dataset_root.resolve()

        # No splits.json for this dataset — there's nothing to hold out
        # from, since this data was never used for training/model-
        # selection at all. Every one of its trials is fair game as test
        # data, using exactly the settings the checkpoint was trained with.
        classes = args.classes if args.classes is not None else ckpt['classes']
        finger = ckpt.get('finger', 'index')
        audio_source = ckpt.get('audio_source', 'surface')
        imu_source = ckpt.get('imu_source', 'fingertip')
        imu_downsample_hz = ckpt.get('imu_downsample_hz')
        print(f'[SETUP] scanning {args.dataset_root}')
        samples = scan_dataset(args.dataset_root, classes, audio_source=audio_source, imu_source=imu_source)
        test_ds = DigitStrokeDataset(samples, finger=finger, audio_source=audio_source, imu_source=imu_source,
                                      watch_downsample_hz=imu_downsample_hz)
        print(f'[SETUP] evaluating on ALL {len(test_ds)} trials under {args.dataset_root} '
              f'(external dataset — not a held-out split) — classes={classes}, '
              f'audio_source="{audio_source}", imu_source="{imu_source}", finger="{finger}"'
              f'{f", imu_downsample_hz={imu_downsample_hz}" if imu_downsample_hz else ""}')
    else:
        splits, classes, finger, audio_source, imu_source, seed = load_splits(args.splits)
        imu_downsample_hz = ckpt.get('imu_downsample_hz')   # not part of splits.json — a preprocessing
                                                              # detail that must match training, so it comes
                                                              # from the checkpoint, same as audio/imu_source
        test_ds = DigitStrokeDataset(splits['test'], finger=finger, audio_source=audio_source,
                                      imu_source=imu_source, watch_downsample_hz=imu_downsample_hz)
        print(f'[SETUP] {len(test_ds)} test trials  classes={classes}  '
              f'audio_source="{audio_source}"  imu_source="{imu_source}"  finger="{finger}"  '
              f'{f"imu_downsample_hz={imu_downsample_hz}  " if imu_downsample_hz else ""}'
              f'(split seed={seed})')
        if ckpt['classes'] != classes:
            print(f'[WARN] checkpoint was trained on classes={ckpt["classes"]}, '
                  f'but this split uses classes={classes} — results will be meaningless '
                  f'if these differ in order or content')
        if ckpt.get('audio_source', 'surface') != audio_source or ckpt.get('imu_source', 'fingertip') != imu_source:
            print(f'[WARN] checkpoint was trained with audio_source="{ckpt.get("audio_source")}", '
                  f'imu_source="{ckpt.get("imu_source")}", but this split uses '
                  f'audio_source="{audio_source}", imu_source="{imu_source}" — the model\'s input '
                  f'distribution won\'t match what it saw during training')

    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    model = build_model(modality, n_classes=len(classes)).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f'[SETUP] loaded checkpoint from epoch {ckpt["epoch"]} (modality="{modality}", '
          f'val_acc={ckpt["val_acc"]:.3f} at train time)')

    test_loss, test_acc, confusion = evaluate(model, modality, test_loader, device, len(classes))
    print(f'\n[RESULT] test_loss={test_loss:.4f}  test_acc={test_acc:.3f}')
    print_report(confusion, classes)

    plot_path = args.plot_out if args.plot_out is not None else args.checkpoint.parent / 'confusion_matrix.png'
    plot_confusion_matrix(confusion, classes, test_acc, modality, plot_path)
    print(f'\n[PLOT] confusion matrix saved to {plot_path}')


if __name__ == '__main__':
    main()