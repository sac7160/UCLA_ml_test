"""
backfill_training_config.py
────────────────────────────────────────────────────────────────────────────
train_ctc.py now auto-saves a training_config.json alongside every
checkpoint it produces (see that file's own comment on this), which
compare_checkpoints.py auto-detects and shows in its comparison table.
Checkpoints trained BEFORE that feature existed don't have this file —
this script recreates it after the fact, from the ORIGINAL command line
you trained that checkpoint with (which you have to remember/have saved
somewhere — this script has no way to recover it from the checkpoint
itself).

Uses the exact same flag NAMES and DEFAULTS as train_ctc.py's own
_KEY_CONFIG_FIELDS subset (see compare_checkpoints.py) — the only fields
that actually end up shown in the comparison table — so you can take your
original training command, drop the parts that aren't in that subset
(--classes, --epochs, --participants-root, --curriculum-stage1-steps,
--out-dir, etc. — none of those are compared/shown), keep the rest
as-is, and just add --checkpoint pointing at the checkpoint this backfill
is for.

Usage — given you trained a checkpoint with:
    python train_ctc.py --participants-root ../dataset --classes a b c ... z \\
        --modality fusion --audio-source watch --imu-source watch \\
        --epochs 200 --synthetic-per-epoch 3000 --concat-letters-min 2 --concat-letters-max 12 \\
        --use-word-trials --use-sentence-trials --sentence-weight 5.0 \\
        --curriculum --curriculum-stage1-steps 5000 --curriculum-stage2-steps 5000 \\
        --motion-loss-weight 0.3 --spec-loss-weight 0.2 --no-space \\
        --out-dir checkpoints/letter_ctc_no_space

you'd backfill its config with:
    python backfill_training_config.py \\
        --checkpoint checkpoints/letter_ctc_no_space/best_model.pt \\
        --modality fusion --audio-source watch --imu-source watch \\
        --synthetic-per-epoch 3000 --sentence-weight 5.0 --curriculum \\
        --motion-loss-weight 0.3 --spec-loss-weight 0.2 --no-space
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description='Recreate training_config.json for a checkpoint trained before that file existed, '
                     'from the original command line you trained it with — see this file\'s own docstring '
                     'for a worked example.')
    parser.add_argument('--checkpoint', type=Path, required=True,
                         help='path to the checkpoint (e.g. .../best_model.pt) this backfill is for — '
                              'training_config.json is written into ITS OWN folder, same convention '
                              'train_ctc.py itself uses (and compare_checkpoints.py auto-detects).')
    # Everything below mirrors train_ctc.py's own flag names/defaults for
    # exactly the subset compare_checkpoints.py's _KEY_CONFIG_FIELDS
    # reads — nothing else actually gets shown in the comparison table,
    # so nothing else needs backfilling here.
    parser.add_argument('--modality', choices=['fusion', 'audio', 'imu'], default='fusion')
    parser.add_argument('--audio-source', default='surface')
    parser.add_argument('--imu-source', default='fingertip')
    parser.add_argument('--sequence-encoder', choices=['gru', 'transformer'], default='gru')
    parser.add_argument('--rnn-hidden', type=int, default=256)
    parser.add_argument('--use-space', action='store_true', default=True)
    parser.add_argument('--no-space', dest='use_space', action='store_false')
    parser.add_argument('--motion-loss-weight', type=float, default=0.0)
    parser.add_argument('--spec-loss-weight', type=float, default=0.0)
    parser.add_argument('--curriculum', action='store_true', default=False)
    parser.add_argument('--sentence-weight', type=float, default=3.0)
    parser.add_argument('--synthetic-per-epoch', type=int, default=0)
    args = parser.parse_args()

    config_dict = {k: (str(v) if isinstance(v, Path) else v)
                   for k, v in vars(args).items() if k != 'checkpoint'}
    out_path = args.checkpoint.parent / 'training_config.json'
    if out_path.exists():
        print(f'[WARN] {out_path} already exists — overwriting with the values given here.')
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    print(f'[DONE] wrote {out_path}')
    print(f'       {config_dict}')


if __name__ == '__main__':
    main()
