"""
260831 저장방식 수정함
train_leave_one_out.py
────────────────────────────────────────────────────────────────────────────
Leave-One-Participant-Out (LOPO) cross-user evaluation — trains N separate
models (N = number of participants), each excluding exactly ONE participant
entirely from training, then evaluates that model ONLY on the held-out
participant's data. This tests whether the system generalizes to a
genuinely NEW user never seen during training at all — a fundamentally
different (and much harder) question than train_ctc.py's own --splits-json
protocol, which only holds out specific TRIALS within a participant who WAS
otherwise used for training (person-DEPENDENT evaluation, not
person-INDEPENDENT).

Each fold shells out to train_ctc.py (via subprocess) with --participants
set to "everyone except the held-out one", so this script never needs
updating when train_ctc.py's own hyperparameter flags change — every flag
you'd normally pass to train_ctc.py (--epochs, --modality,
--motion-loss-weight, --sequence-encoder, ...) is simply passed straight
through here too. Evaluation reuses evaluate_word_ctc.py's run_evaluation()
directly (no subprocess needed there), restricted to just the held-out
participant, with NO --splits-json — none of their trials were ever used
for training, so every one of them is genuinely held out.

This is a genuinely expensive operation (N full training runs) — meant to
be run once you're already confident in your hyperparameters (see this
project's own model_improvement_plan.md, Phase 5.3), not for everyday
experimentation.

Usage:
    python train_leave_one_out.py \\
        --participants-root ../dataset \\
        --out-dir checkpoints/lopo \\
        --classes a b c ... z --modality fusion --audio-source watch --imu-source watch \\
        --epochs 200 --synthetic-per-epoch 3000 --concat-letters-min 2 --concat-letters-max 12 \\
        --use-word-trials --use-sentence-trials --sentence-weight 5.0 \\
        --curriculum --curriculum-stage1-steps 5000 --curriculum-stage2-steps 5000 \\
        --motion-loss-weight 0.3 --spec-loss-weight 0.2 \\
        --sequence-encoder transformer --transformer-layers 2 --transformer-nhead 8

    # Every flag after the LOPO-specific ones above (--epochs onward) is
    # passed straight through to train_ctc.py for every fold, unmodified.

    # To run only a subset of folds (e.g. resuming after a crash, or
    # spreading folds across machines):
        --held-out-participants p3 p7 p12
"""

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

# This file lives in the CTC subfolder (e.g. 260829test/), alongside
# dataset_ctc.py/evaluate_word_ctc.py — but this file itself does
# `from dataset_ctc import ...` below, and dataset_ctc.py in turn does
# `import config_ctc`, which does `import config` (the ORIGINAL,
# unprefixed config.py that lives one level up, in ml/ itself). Python
# only auto-adds THIS script's own directory to sys.path, never its
# parent, so without inserting the grandparent directory too, that
# import chain fails with "No module named 'config'" the moment this
# script is run from anywhere other than exactly the right cwd — the
# same fix train_ctc.py itself applies to its own sys.path, needed here
# too since this file imports dataset_ctc directly (not just via
# subprocess-invoking train_ctc.py, which fixes its own sys.path
# independently once it starts).
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset_ctc import discover_participant_dataset_dirs
from evaluate_word_ctc import run_evaluation, _cer, _wer


def _summarize(df: pd.DataFrame, target: str, has_dictionary: bool) -> dict:
    """Same aggregate metrics evaluate_word_ctc.py's _print_summary_block
    prints, flattened into one row's worth of a LOPO fold's results — see
    that file's own _cer/_wer docstrings for why CER/WER (not just
    exact-match accuracy) matter for judging how close a decode actually
    got."""
    sub = df[df['writing_target'] == target]
    if len(sub) == 0:
        return {f'{target}_n': 0}
    out = {
        f'{target}_n': len(sub),
        f'{target}_raw_accuracy': round(sub['raw_correct'].mean(), 3),
        f'{target}_raw_cer': round(_cer(sub, 'raw_ctc_pred', 'raw_edit_distance'), 3),
        f'{target}_raw_wer': round(_wer(sub, 'raw_ctc_pred'), 3),
    }
    if has_dictionary and 'dict_correct' in sub.columns:
        out[f'{target}_corrected_accuracy'] = round(sub['dict_correct'].mean(), 3)
        out[f'{target}_corrected_cer'] = round(_cer(sub, 'dict_snapped_pred', 'dict_edit_distance'), 3)
        out[f'{target}_corrected_wer'] = round(_wer(sub, 'dict_snapped_pred'), 3)
    return out


def _write_summary_and_aggregate(summary_rows: list, out_dir: Path):
    """Builds and saves lopo_summary.csv (one row per fold completed SO
    FAR) and lopo_aggregate.csv (mean ± std across those folds), then
    returns both as DataFrames for the caller to print. Called after
    EVERY fold (not just once at the very end) so an interrupted run
    still leaves a usable, up-to-date summary behind — see the call
    site's own comment for why this matters for a long-running,
    multi-fold operation like LOPO."""
    summary = pd.DataFrame(summary_rows).set_index('held_out_participant')
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / 'lopo_summary.csv')

    # The headline "does this generalize to a new user" number — mean and
    # spread (std) across folds. A LARGE std here (relative to the mean)
    # is itself an important finding: it means generalization quality
    # depends heavily on WHICH user is held out, not just the sensing
    # modality/architecture choices. With only 1 fold done so far, std
    # is undefined (NaN) — expected, not a bug; it fills in as more
    # folds complete.
    numeric_cols = summary.select_dtypes(include='number').columns
    agg = summary[numeric_cols].agg(['mean', 'std'])
    agg.to_csv(out_dir / 'lopo_aggregate.csv')
    return summary, agg


def main():
    parser = argparse.ArgumentParser(
        description='Leave-One-Participant-Out cross-user evaluation — trains N models (one per '
                     'held-out participant) and reports how each generalizes to a completely unseen user.')
    parser.add_argument('--participants-root', type=Path, required=True,
                         help='same meaning as train_ctc.py/evaluate_word_ctc.py\'s own '
                              '--participants-root — every "p*"-named folder found here is a fold '
                              'candidate unless --held-out-participants restricts to a subset.')
    parser.add_argument('--held-out-participants', nargs='+', default=None,
                         help='which participants to run as the held-out fold — default: every '
                              'participant found under --participants-root (a full LOPO sweep). Restrict '
                              'this to resume after a partial run, or to split folds across machines.')
    parser.add_argument('--out-dir', type=Path, default=Path('checkpoints/lopo'),
                         help='each fold\'s checkpoint/results land in <out-dir>/<held_out_participant>/; '
                              'the cross-fold summary lands directly in <out-dir>/.')
    parser.add_argument('--include-sentences', action='store_true')
    parser.add_argument('--phrase-set-path', type=Path, default=None)
    parser.add_argument('--decode', choices=['greedy', 'beam'], default='greedy')
    parser.add_argument('--beam-width', type=int, default=10)
    parser.add_argument('--skip-training', action='store_true',
                         help='skip the training subprocess entirely and only run evaluation — for '
                              're-scoring folds whose checkpoints already exist (e.g. after changing '
                              '--decode/--phrase-set-path) without re-training them.')
    # Every OTHER flag (--epochs, --modality, --motion-loss-weight, ...) is
    # passed straight through to train_ctc.py for every fold, unmodified —
    # this script deliberately does NOT re-declare train_ctc.py's dozens of
    # hyperparameter flags, so it never goes stale when those change.
    args, train_passthrough_args = parser.parse_known_args()

    all_participants = sorted(name for name, _ in discover_participant_dataset_dirs(args.participants_root))
    if not all_participants:
        raise RuntimeError(f'no participant folders found under {args.participants_root}')
    held_out_list = args.held_out_participants or all_participants
    print(f'[LOPO] {len(all_participants)} participants found; running {len(held_out_list)} fold(s): '
          f'{held_out_list}')

    train_script = Path(__file__).resolve().parent / 'train_ctc.py'
    summary_rows = []

    for held_out in held_out_list:
        if held_out not in all_participants:
            print(f'[LOPO] WARNING: "{held_out}" not found under {args.participants_root} — skipping')
            continue
        train_participants = [p for p in all_participants if p != held_out]
        fold_out_dir = args.out_dir / held_out
        print(f'\n{"=" * 70}\n[LOPO] fold: holding out "{held_out}" '
              f'({len(train_participants)} participants for training)\n{"=" * 70}')

        checkpoint = fold_out_dir / 'best_model.pt'
        if not args.skip_training:
            # --participants explicitly lists everyone EXCEPT the held-out
            # one, so that participant contributes ZERO trials anywhere in
            # training/validation — the strongest possible holdout, unlike
            # --splits-json (which only excludes specific trials within a
            # participant who otherwise WAS trained on).
            train_cmd = [
                sys.executable, str(train_script),
                '--participants-root', str(args.participants_root),
                '--participants', *train_participants,
                '--out-dir', str(fold_out_dir),
            ] + train_passthrough_args
            print(f'[LOPO] training: {" ".join(train_cmd)}')
            result = subprocess.run(train_cmd)
            if result.returncode != 0:
                print(f'[LOPO] WARNING: training failed for held-out "{held_out}" '
                      f'(exit code {result.returncode}) — skipping evaluation for this fold')
                continue
        elif not checkpoint.exists():
            print(f'[LOPO] WARNING: --skip-training given but no checkpoint at {checkpoint} — skipping')
            continue

        if not checkpoint.exists():
            print(f'[LOPO] WARNING: no checkpoint produced for held-out "{held_out}" — skipping')
            continue

        # Evaluate ONLY on the held-out participant — no --splits-json
        # needed here: this participant contributed nothing to training,
        # so every one of their trials is genuinely held out.
        eval_args = argparse.Namespace(
            checkpoint=checkpoint, participants_root=args.participants_root,
            participants=[held_out], word_subfolder='word', sentence_subfolder='sentence',
            include_sentences=args.include_sentences, splits_json=None,
            phrase_set_path=args.phrase_set_path, decode=args.decode, beam_width=args.beam_width,
            quiet=True, out_dir=fold_out_dir,
        )
        df = run_evaluation(eval_args)
        fold_out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(fold_out_dir / 'word_ctc_results.csv', index=False)

        has_dict = 'dict_correct' in df.columns
        row = {'held_out_participant': held_out, 'n_train_participants': len(train_participants)}
        row.update(_summarize(df, 'word', has_dict))
        row.update(_summarize(df, 'sentence', has_dict))
        summary_rows.append(row)
        print(f'[LOPO] fold "{held_out}" done — '
              f'word_raw_accuracy={row.get("word_raw_accuracy", float("nan"))}, '
              f'word_raw_cer={row.get("word_raw_cer", float("nan"))}')

        # Re-written after EVERY fold, not just once at the very end —
        # LOPO is a long-running, multi-hour (or multi-day) operation
        # split across N full training runs; if the process is
        # interrupted (crash, forced reboot — see this project's own
        # earlier system-stability issues) partway through, whatever
        # folds DID finish should still leave a usable summary/aggregate
        # behind, not nothing. Cheap to redo each time (a few KB of CSV).
        _write_summary_and_aggregate(summary_rows, args.out_dir)

    if not summary_rows:
        raise RuntimeError('no fold completed successfully — nothing to summarize')

    summary, agg = _write_summary_and_aggregate(summary_rows, args.out_dir)
    print(f'\n\n{"#" * 70}\n[RESULT] Leave-One-Participant-Out summary across {len(summary)} fold(s)'
          f'\n{"#" * 70}\n')
    with pd.option_context('display.max_columns', None, 'display.width', 200):
        print(summary.to_string())
    print('\n[RESULT] aggregate across folds (mean ± std — this is the cross-user generalization number):')
    with pd.option_context('display.max_columns', None, 'display.width', 200):
        print(agg.to_string())
    print(f'\n[DATA] {args.out_dir}/lopo_summary.csv  (per-fold)')
    print(f'[DATA] {args.out_dir}/lopo_aggregate.csv  (mean ± std across folds)')


if __name__ == '__main__':
    main()