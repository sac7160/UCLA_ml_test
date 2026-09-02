"""
evaluate_by_surface.py
────────────────────────────────────────────────────────────────────────────
Breaks down an already-trained model's accuracy/CER by writing SURFACE
(trial_info.json's "material" field — e.g. "wood", "desk", whatever
values this project's own collection actually used) — a
finer-grained slice of run_evaluation's own results, not a separate
evaluation pipeline: this script calls evaluate_word_ctc.run_evaluation()
exactly the way evaluate_word_ctc.py itself does, then re-groups the SAME
per-trial rows by material instead of (or alongside) by writing_target
(letter/word/sentence).

"material" lives in each trial's own trial_info.json (see this project's
own sample:
    {"participant": "p1", "label": "a", "content": "a", "material": "wood", ...}
) — NOT in the checkpoint or splits.json, so this script re-opens each
evaluated trial's trial_info.json individually after run_evaluation()
returns, to attach that trial's material as an extra column.

Usage:
    python evaluate_by_surface.py \
        --checkpoint checkpoints/gru_lopo/best_model.pt \
        --participants-root ../dataset \
        --splits-json checkpoints/gru_lopo/splits.json \
        --include-sentences --include-letters
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluate_word_ctc import run_evaluation, _cer, _wer


def _load_material(trial_dir: str) -> str:
    """Reads just the "material" field out of one trial's trial_info.json
    — returns "(unknown)" (not None/NaN) when the file is missing that
    field, or missing entirely (e.g. a LETTER trial evaluated via
    --include-letters may predate this field being recorded, or a
    participant's data collection used an older trial_info.json schema)
    — this way a missing-material trial still shows up as its own
    explicit group in the report, rather than silently vanishing from
    a groupby."""
    info_path = Path(trial_dir) / 'trial_info.json'
    if not info_path.exists():
        return '(unknown)'
    try:
        with open(info_path) as f:
            info = json.load(f)
    except (json.JSONDecodeError, OSError):
        return '(unknown)'
    return info.get('material') or '(unknown)'


def _summarize_group(df: pd.DataFrame, has_dictionary: bool) -> dict:
    """Same metrics evaluate_word_ctc.py's own _summarize_metrics
    computes, for one (writing_target, material) slice."""
    out = {
        'n': int(len(df)),
        'raw_accuracy': round(float(df['raw_correct'].mean()), 4),
        'raw_cer': round(float(_cer(df, 'raw_ctc_pred', 'raw_edit_distance')), 4),
    }
    wer = _wer(df, 'raw_ctc_pred')
    out['raw_wer'] = round(float(wer), 4) if not pd.isna(wer) else None
    if has_dictionary and 'dict_correct' in df.columns:
        out['corrected_accuracy'] = round(float(df['dict_correct'].mean()), 4)
        out['corrected_cer'] = round(float(_cer(df, 'dict_snapped_pred', 'dict_edit_distance')), 4)
        corrected_wer = _wer(df, 'dict_snapped_pred')
        out['corrected_wer'] = round(float(corrected_wer), 4) if not pd.isna(corrected_wer) else None
    return out


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description='Breaks down evaluate_word_ctc.py\'s own results by writing surface '
                     '(trial_info.json\'s "material" field) — same checkpoint, same trials, just '
                     're-grouped by material instead of only by writing_target.')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--participants-root', type=Path, required=True)
    parser.add_argument('--participants', nargs='+', default=None)
    parser.add_argument('--word-subfolder', default='word')
    parser.add_argument('--sentence-subfolder', default='sentence')
    parser.add_argument('--include-sentences', action='store_true')
    parser.add_argument('--include-letters', action='store_true')
    parser.add_argument('--splits-json', type=Path, default=None)
    parser.add_argument('--phrase-set-path', type=Path, default=None)
    parser.add_argument('--decode', choices=['greedy', 'beam'], default='greedy')
    parser.add_argument('--beam-width', type=int, default=10)
    parser.add_argument('--out-dir', type=Path, default=Path('surface_eval'))
    return parser


def main():
    args = build_arg_parser().parse_args()
    args.quiet = True   # this script prints its OWN per-surface summary instead — run_evaluation's
                          # own per-trial [   ][W] lines would just be noise here
    df = run_evaluation(args)

    df['material'] = df['trial'].apply(_load_material)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dir / 'per_trial_with_material.csv', index=False)

    has_dictionary = 'dict_correct' in df.columns
    materials = sorted(df['material'].unique())
    print(f'\n[RESULT] materials found: {materials}')

    report = {'checkpoint': str(args.checkpoint), 'materials': {}}
    summary_rows = []
    for target in ('letter', 'word', 'sentence'):
        target_df = df[df['writing_target'] == target]
        if len(target_df) == 0:
            continue
        report['materials'].setdefault(target, {})
        print(f'\n[RESULT] === {target.upper()} by surface ===')
        for material in materials:
            sub = target_df[target_df['material'] == material]
            if len(sub) == 0:
                continue
            metrics = _summarize_group(sub, has_dictionary)
            report['materials'][target][material] = metrics
            summary_rows.append({'writing_target': target, 'material': material, **metrics})
            print(f'  {material:12s}  n={metrics["n"]:4d}  '
                  f'accuracy={metrics["raw_accuracy"]:.3f}  cer={metrics["raw_cer"]:.3f}'
                  + (f'  wer={metrics["raw_wer"]:.3f}' if metrics['raw_wer'] is not None else ''))

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(args.out_dir / 'summary_by_surface.csv', index=False)
    with open(args.out_dir / 'report_by_surface.json', 'w') as f:
        json.dump(report, f, indent=2)

    print(f'\n[DATA] {args.out_dir}/per_trial_with_material.csv  (every trial, material attached)')
    print(f'[DATA] {args.out_dir}/summary_by_surface.csv  (one row per writing_target x material)')
    print(f'[DATA] {args.out_dir}/report_by_surface.json  (same, nested JSON)')


if __name__ == '__main__':
    main()
