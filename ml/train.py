"""
digit_classifier/train.py
────────────────────────────────────────────────────────────────────────────
Trains DigitFusionNet on the train split, model-selects on val, and never
touches test — test.py is a separate script, run once, after this.

Paths (--dataset-root, --out-dir) default to locations relative to *this
script's own directory* (via SCRIPT_DIR below), not the shell's current
directory — so `python train.py` and `python ml/train.py` from one level up
both land in the same place regardless of where you happen to run them from.

Usage:
    python train.py --epochs 30
    python train.py --audio-source watch --imu-source watch --out-dir checkpoints/watch_pilot
    python train.py --classes digits_0 digits_1 digits_2 --out-dir checkpoints/pilot1
"""

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from dataset import DigitStrokeDataset, scan_dataset, make_splits, save_splits
from model import build_model, forward_model, MODALITIES

SCRIPT_DIR = Path(__file__).resolve().parent


def run_epoch(model, modality, loader, device, optimizer=None, desc: str = ''):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss, n_correct, n_total = 0.0, 0, 0

    # leave=False: the bar clears itself once the epoch finishes, so 30
    # epochs' worth of bars don't pile up in the terminal — only the
    # [EPOCH ...] summary line printed after this stays behind.
    pbar = tqdm(loader, desc=desc, leave=False, ncols=100)
    with torch.set_grad_enabled(is_train):
        for audio, imu, label in pbar:
            audio, imu, label = audio.to(device), imu.to(device), label.to(device)
            logits = forward_model(model, modality, audio, imu)
            loss = F.cross_entropy(logits, label)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * label.size(0)
            n_correct += (logits.argmax(dim=1) == label).sum().item()
            n_total += label.size(0)
            pbar.set_postfix(loss=f'{total_loss / n_total:.4f}', acc=f'{n_correct / n_total:.3f}')

    return total_loss / n_total, n_correct / n_total


def plot_training_curves(history: dict, modality: str, best_epoch: int, save_path: Path):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax_loss.plot(epochs, history['train_loss'], label='train', color='#1f77b4')
    ax_loss.plot(epochs, history['val_loss'], label='val', color='#d62728')
    ax_loss.axvline(best_epoch, color='#888', linestyle='--', linewidth=1, label='best val_acc')
    ax_loss.set_xlabel('epoch'); ax_loss.set_ylabel('loss'); ax_loss.set_title('Loss')
    ax_loss.legend(); ax_loss.grid(alpha=0.3)

    ax_acc.plot(epochs, history['train_acc'], label='train', color='#1f77b4')
    ax_acc.plot(epochs, history['val_acc'], label='val', color='#d62728')
    ax_acc.axvline(best_epoch, color='#888', linestyle='--', linewidth=1, label='best val_acc')
    ax_acc.set_xlabel('epoch'); ax_acc.set_ylabel('accuracy'); ax_acc.set_ylim(0, 1.05)
    ax_acc.set_title('Accuracy')
    ax_acc.legend(); ax_acc.grid(alpha=0.3)

    fig.suptitle(f'Training curves ({modality})')
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Train the pilot digit classifier')
    parser.add_argument('--dataset-root', type=Path, default=SCRIPT_DIR.parent / 'dataset',
                         help='defaults to run/dataset, i.e. one level up from this script')
    parser.add_argument('--classes', nargs='+', default=config.DEFAULT_CLASSES)
    parser.add_argument('--modality', choices=list(MODALITIES), default='fusion',
                         help='"fusion" = audio+IMU, "audio" = audio only, "imu" = IMU only — '
                              'lets you compare the fused model against each single modality')
    parser.add_argument('--audio-source', choices=list(config.AUDIO_FILENAMES), default=config.DEFAULT_AUDIO_SOURCE,
                         help='"surface" = surface_mic.wav, "watch" = watch_audio.wav')
    parser.add_argument('--imu-source', choices=list(config.IMU_FILENAMES), default=config.DEFAULT_IMU_SOURCE,
                         help='"fingertip" = fingertip_imu.csv (needs --finger), "watch" = imu.csv')
    parser.add_argument('--finger', default=config.DEFAULT_FINGER, help='only used when --imu-source fingertip')
    parser.add_argument('--imu-downsample-hz', type=float, default=None,
                         help='only used when --imu-source watch — simulates the watch sensor having only '
                              'sampled at this rate (nearest-neighbor, not interpolation), to test whether '
                              'temporal resolution explains an accuracy gap vs. fingertip_imu (which natively '
                              'runs at camera frame rate, ~20-30Hz)')
    parser.add_argument('--train-frac', type=float, default=0.7)
    parser.add_argument('--val-frac', type=float, default=0.15)
    parser.add_argument('--test-frac', type=float, default=0.15)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out-dir', type=Path, default=SCRIPT_DIR / 'checkpoints',
                         help='where to save splits.json + best_model.pt. Defaults to ml/checkpoints — '
                              'if you run more than one experiment config (e.g. different '
                              '--modality/--audio-source/--imu-source), pass a distinct --out-dir '
                              'for each, or later runs will overwrite earlier ones\' checkpoint + splits.')
    args = parser.parse_args()

    if args.imu_downsample_hz is not None and args.imu_source != 'watch':
        parser.error('--imu-downsample-hz only applies when --imu-source watch')

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[SETUP] device={device}')

    samples = scan_dataset(args.dataset_root, args.classes,
                            audio_source=args.audio_source, imu_source=args.imu_source)
    splits = make_splits(samples, args.train_frac, args.val_frac, args.test_frac, args.seed)
    splits_path = args.out_dir / 'splits.json'
    save_splits(splits, splits_path, args.classes, args.finger, args.seed,
                audio_source=args.audio_source, imu_source=args.imu_source)
    print(f'[SETUP] {len(splits["train"])} train / {len(splits["val"])} val / '
          f'{len(splits["test"])} test trials  (classes={args.classes}, modality="{args.modality}", '
          f'audio_source="{args.audio_source}", imu_source="{args.imu_source}", finger="{args.finger}"'
          f'{f", imu_downsample_hz={args.imu_downsample_hz}" if args.imu_downsample_hz else ""})')
    print(f'[SETUP] split saved to {splits_path} — test.py will load this exact split, '
          f'so the test set stays held-out')

    train_ds = DigitStrokeDataset(splits['train'], finger=args.finger, audio_source=args.audio_source,
                                   imu_source=args.imu_source, watch_downsample_hz=args.imu_downsample_hz)
    val_ds = DigitStrokeDataset(splits['val'], finger=args.finger, audio_source=args.audio_source,
                                 imu_source=args.imu_source, watch_downsample_hz=args.imu_downsample_hz)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = build_model(args.modality, n_classes=len(args.classes)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = -1.0
    best_epoch = 0
    ckpt_path = args.out_dir / 'best_model.pt'
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, args.modality, train_loader, device, optimizer,
                                           desc=f'epoch {epoch:3d}/{args.epochs} [train]')
        val_loss, val_acc = run_epoch(model, args.modality, val_loader, device, optimizer=None,
                                       desc=f'epoch {epoch:3d}/{args.epochs} [val]  ')
        print(f'[EPOCH {epoch:3d}/{args.epochs}] '
              f'train_loss={train_loss:.4f} train_acc={train_acc:.3f}  '
              f'val_loss={val_loss:.4f} val_acc={val_acc:.3f}')
        history['train_loss'].append(train_loss); history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss); history['val_acc'].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            # Re-create right before every save, not just once at startup —
            # cloud-synced folders (iCloud Desktop/Documents sync is the
            # common case on macOS) can evict/remove a directory mid-run
            # without the script doing anything wrong; recreating here is
            # cheap (no-op if it's already there) and means a save can't
            # crash the whole run over something outside this script's
            # control.
            args.out_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'classes': args.classes,
                'modality': args.modality,
                'finger': args.finger,
                'audio_source': args.audio_source,
                'imu_source': args.imu_source,
                'imu_downsample_hz': args.imu_downsample_hz,
                'val_acc': val_acc,
                'epoch': epoch,
            }, ckpt_path)

    args.out_dir.mkdir(parents=True, exist_ok=True)   # same defensive re-check before the final plot/history writes

    curves_path = args.out_dir / 'training_curves.png'
    plot_training_curves(history, args.modality, best_epoch, curves_path)
    with open(args.out_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f'\n[DONE] best val_acc={best_val_acc:.3f} (epoch {best_epoch}) — checkpoint: {ckpt_path}')
    print(f'[PLOT] training curves saved to {curves_path}')
    print(f'[NEXT] python test.py --checkpoint {ckpt_path} --splits {splits_path}')


if __name__ == '__main__':
    main()