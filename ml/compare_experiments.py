"""
digit_classifier/compare_experiments.py
────────────────────────────────────────────────────────────────────────────
Summarizes several already-trained experiments side by side, for exactly
the "explain the results to someone else at a glance" situation: each
experiment is its own checkpoints/<name>/ folder (produced by train.py)
containing best_model.pt + splits.json (+ history.json, if you want the
validation curves included too).

Re-evaluates every experiment's held-out test set fresh (reusing test.py's
own evaluate() function, so the numbers are computed identically to
running test.py by hand on each one) rather than trusting anything
pre-saved — so this is always in sync with whatever's actually in
checkpoints/, no matter which experiments you've re-run since.

Usage:
    python compare_experiments.py
    python compare_experiments.py --checkpoints-dir checkpoints
    python compare_experiments.py --experiments checkpoints/surface_only checkpoints/watchmic_only
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from dataset import DigitStrokeDataset, load_splits
from model import build_model
from test import evaluate

SCRIPT_DIR = Path(__file__).resolve().parent


def discover_experiments(checkpoints_dir: Path) -> list:
    """Every immediate subfolder of checkpoints_dir that has both
    best_model.pt and splits.json — i.e. everything train.py has ever
    written a complete result to."""
    if not checkpoints_dir.exists():
        return []
    return sorted(
        d for d in checkpoints_dir.iterdir()
        if d.is_dir() and (d / 'best_model.pt').exists() and (d / 'splits.json').exists()
    )


def evaluate_experiment(exp_dir: Path, device, batch_size: int) -> dict:
    splits, classes, finger, audio_source, imu_source, seed = load_splits(exp_dir / 'splits.json')
    test_ds = DigitStrokeDataset(splits['test'], finger=finger,
                                  audio_source=audio_source, imu_source=imu_source)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    ckpt = torch.load(exp_dir / 'best_model.pt', map_location=device, weights_only=False)
    modality = ckpt.get('modality', 'fusion')
    model = build_model(modality, n_classes=len(classes)).to(device)
    model.load_state_dict(ckpt['model_state_dict'])

    test_loss, test_acc, confusion = evaluate(model, modality, test_loader, device, len(classes))

    history = None
    history_path = exp_dir / 'history.json'
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)

    return {
        'name': exp_dir.name,
        'modality': modality,
        'audio_source': audio_source,
        'imu_source': imu_source,
        'test_acc': test_acc,
        'test_loss': test_loss,
        'val_acc_best': ckpt.get('val_acc'),
        'n_test': len(test_ds),
        'classes': classes,
        'history': history,
    }


def plot_summary(results: list, save_path: Path):
    # Sorted by test accuracy, best first — for a bar chart meant to be
    # read at a glance, this matters more than alphabetical/insertion order.
    ordered = sorted(results, key=lambda r: -r['test_acc'])
    colors = plt.cm.tab10.colors
    color_by_name = {r['name']: colors[i % 10] for i, r in enumerate(ordered)}

    fig, (ax_bar, ax_curve) = plt.subplots(1, 2, figsize=(max(11, 1.3 * len(results) + 5), 5))

    names = [r['name'] for r in ordered]
    accs = [r['test_acc'] for r in ordered]
    bars = ax_bar.bar(names, accs, color=[color_by_name[n] for n in names])
    ax_bar.set_ylabel('test accuracy')
    ax_bar.set_ylim(0, 1.05)
    ax_bar.set_title('Test accuracy by experiment')
    ax_bar.tick_params(axis='x', rotation=30)
    for label in ax_bar.get_xticklabels():
        label.set_ha('right')
    for bar, acc, r in zip(bars, accs, ordered):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, acc + 0.02, f'{acc:.2f}',
                    ha='center', fontsize=9)
    ax_bar.grid(axis='y', alpha=0.3)

    any_history = False
    for r in ordered:
        if r['history'] is not None:
            any_history = True
            epochs = range(1, len(r['history']['val_acc']) + 1)
            ax_curve.plot(epochs, r['history']['val_acc'], label=r['name'], color=color_by_name[r['name']])
    ax_curve.set_xlabel('epoch'); ax_curve.set_ylabel('val accuracy')
    ax_curve.set_ylim(0, 1.05)
    ax_curve.set_title('Validation accuracy during training')
    if any_history:
        ax_curve.legend(fontsize=8, loc='lower right')
    else:
        ax_curve.text(0.5, 0.5, 'no history.json found for any experiment',
                      ha='center', va='center', transform=ax_curve.transAxes, color='#888')
    ax_curve.grid(alpha=0.3)

    fig.suptitle('Modality / input-source comparison')
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def print_table(results: list):
    ordered = sorted(results, key=lambda r: -r['test_acc'])
    print(f'\n{"experiment":30s} {"modality":8s} {"audio":9s} {"imu":10s} {"test_acc":>9s} {"n_test":>7s}')
    print('-' * 78)
    for r in ordered:
        print(f'{r["name"]:30s} {r["modality"]:8s} {r["audio_source"]:9s} {r["imu_source"]:10s} '
              f'{r["test_acc"]:9.3f} {r["n_test"]:7d}')


def main():
    parser = argparse.ArgumentParser(description='Compare several trained experiments side by side')
    parser.add_argument('--checkpoints-dir', type=Path, default=SCRIPT_DIR / 'checkpoints',
                         help='auto-discovers every subfolder with best_model.pt + splits.json')
    parser.add_argument('--experiments', nargs='+', type=Path, default=None,
                         help='explicit list of experiment folders — overrides --checkpoints-dir '
                              'auto-discovery if given')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--out', type=Path, default=None,
                         help='defaults to <checkpoints-dir>/comparison.png')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    exp_dirs = args.experiments if args.experiments is not None else discover_experiments(args.checkpoints_dir)
    if not exp_dirs:
        raise RuntimeError(f'No experiment folders (with best_model.pt + splits.json) found under '
                            f'{args.checkpoints_dir}. Pass --experiments explicitly if they live elsewhere.')
    print(f'[SETUP] comparing {len(exp_dirs)} experiments: {[d.name for d in exp_dirs]}')

    results = []
    for exp_dir in exp_dirs:
        print(f'[EVAL] {exp_dir.name} ...')
        try:
            results.append(evaluate_experiment(exp_dir, device, args.batch_size))
        except Exception as e:
            print(f'[WARN] skipping {exp_dir.name}: {e}')

    if not results:
        raise RuntimeError('None of the discovered experiments could be evaluated — see warnings above')

    print_table(results)

    out_path = args.out if args.out is not None else args.checkpoints_dir / 'comparison.png'
    plot_summary(results, out_path)
    print(f'\n[PLOT] comparison saved to {out_path}')

    summary_path = out_path.with_suffix('.json')
    with open(summary_path, 'w') as f:
        json.dump([{k: v for k, v in r.items() if k != 'history'} for r in results], f, indent=2)
    print(f'[DATA] summary numbers saved to {summary_path}')


if __name__ == '__main__':
    main()