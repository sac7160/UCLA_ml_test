"""
compare_checkpoints.py
────────────────────────────────────────────────────────────────────────────
Runs evaluate_word_ctc.py's evaluation logic across SEVERAL checkpoints in
one go, and builds a single side-by-side comparison table + bar chart.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# 현재 경로(ml 폴더) 및 260821test 폴더를 sys.path에 추가하여 모듈 참조 가능하게 설정
current_dir = Path(__file__).resolve().parent
target_dir = current_dir / "260821test"

if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))
if str(target_dir) not in sys.path:
    sys.path.insert(0, str(target_dir))

import matplotlib
matplotlib.use('Agg')   # GUI 창 팝업 방지 (터미널 환경 실행)
import matplotlib.pyplot as plt
import pandas as pd

# 260821test/evaluate_word_ctc.py 모듈 불러오기
import evaluate_word_ctc


def run_evaluation(eval_args):
    """
    evaluate_word_ctc.py의 main()을 인자 오버라이딩을 통해 실행하고
    생성된 결과 DataFrame을 읽어서 반환하는 래퍼 함수
    """
    cmd_args = [
        'evaluate_word_ctc.py',
        '--checkpoint', str(eval_args.checkpoint),
        '--out-dir', str(eval_args.out_dir),
        '--word-subfolder', str(eval_args.word_subfolder),
        '--sentence-subfolder', str(eval_args.sentence_subfolder),
        '--decode', str(eval_args.decode),
        '--beam-width', str(eval_args.beam_width),
    ]
    if eval_args.participants_root:
        cmd_args.extend(['--participants-root', str(eval_args.participants_root)])
    if eval_args.participants:
        cmd_args.extend(['--participants'] + list(eval_args.participants))
    if eval_args.include_sentences:
        cmd_args.append('--include-sentences')
    if eval_args.splits_json:
        cmd_args.extend(['--splits-json', str(eval_args.splits_json)])
    if eval_args.phrase_set_path:
        cmd_args.extend(['--phrase-set-path', str(eval_args.phrase_set_path)])
    if getattr(eval_args, 'quiet', False):
        cmd_args.append('--quiet')

    # sys.argv 임시 조작 후 evaluate_word_ctc.main() 실행
    old_argv = sys.argv
    try:
        sys.argv = cmd_args
        evaluate_word_ctc.main()
    finally:
        sys.argv = old_argv

    res_csv = eval_args.out_dir / 'word_ctc_results.csv'
    if not res_csv.exists():
        raise FileNotFoundError(f"Evaluation failed: {res_csv} 파일이 생성되지 않았습니다.")

    # 빈 예측값("")이 NaN으로 파싱되는 현상을 막기 위해 keep_default_na=False 사용
    return pd.read_csv(res_csv, keep_default_na=False)


_KEY_CONFIG_FIELDS = [
    'modality', 'audio_source', 'imu_source', 'sequence_encoder', 'use_space',
    'motion_loss_weight', 'spec_loss_weight', 'curriculum', 'sentence_weight',
    'synthetic_per_epoch', 'rnn_hidden',
]


def _load_training_config(checkpoint_path: Path) -> dict:
    candidate = checkpoint_path.parent / 'training_config.json'
    if not candidate.exists():
        return {}
    with open(candidate) as f:
        full_config = json.load(f)
    return {k: full_config.get(k) for k in _KEY_CONFIG_FIELDS if k in full_config}


def _parse_experiment(spec: str):
    if '=' not in spec:
        raise argparse.ArgumentTypeError(
            f'--experiments entries must look like LABEL=PATH, got "{spec}" (no "=" found)')
    label, path_str = spec.split('=', 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError(f'--experiments entry "{spec}" has an empty label before "="')
    return label, Path(path_str.strip())


def _word_edit_distance(a_words: list, b_words: list) -> int:
    dp = list(range(len(b_words) + 1))
    for i, wa in enumerate(a_words, 1):
        prev, dp[0] = dp[0], i
        for j, wb in enumerate(b_words, 1):
            prev, dp[j] = dp[j], min(dp[j] + 1, dp[j - 1] + 1, prev + (wa != wb))
    return dp[-1]


def _cer(df: pd.DataFrame, pred_col: str, ed_col: str) -> float:
    total_chars = df['true_word'].astype(str).str.len().sum()
    if total_chars == 0:
        return float('nan')
    return df[ed_col].sum() / total_chars


def _wer(df: pd.DataFrame, pred_col: str) -> float:
    if not df['true_word'].astype(str).str.contains(' ').any():
        return float('nan')
    total_words = 0
    total_word_edits = 0
    for true_text, pred_text in zip(df['true_word'].astype(str), df[pred_col].astype(str)):
        true_words = true_text.split()
        pred_words = pred_text.split()
        total_words += len(true_words)
        total_word_edits += _word_edit_distance(true_words, pred_words)
    return total_word_edits / total_words if total_words else float('nan')


def _summarize(df: pd.DataFrame, target: str, has_dictionary: bool) -> dict:
    sub = df[df['writing_target'] == target]
    if len(sub) == 0:
        return {f'{target}_n': 0}
    out = {
        f'{target}_n': len(sub),
        f'{target}_raw_accuracy': round(sub['raw_correct'].mean(), 3),
        f'{target}_raw_cer': round(_cer(sub, 'raw_ctc_pred', 'raw_edit_distance'), 3),
        f'{target}_raw_wer': round(_wer(sub, 'raw_ctc_pred'), 3),
        f'{target}_blank_pct': round((sub['raw_ctc_pred'] == '').mean() * 100, 1),
    }
    if has_dictionary and 'dict_correct' in sub.columns:
        out[f'{target}_corrected_accuracy'] = round(sub['dict_correct'].mean(), 3)
        out[f'{target}_corrected_cer'] = round(_cer(sub, 'dict_snapped_pred', 'dict_edit_distance'), 3)
        out[f'{target}_corrected_wer'] = round(_wer(sub, 'dict_snapped_pred'), 3)
    return out


def main():
    parser = argparse.ArgumentParser(description='Evaluate several checkpoints on the same held-out data '
                                                  'and build one side-by-side comparison table')
    parser.add_argument('--experiments', nargs='+', required=True, type=_parse_experiment,
                         help='one or more LABEL=CHECKPOINT_PATH pairs')
    parser.add_argument('--participants-root', type=Path,
                         default=Path(__file__).resolve().parent.parent.parent / 'dataset')
    parser.add_argument('--participants', nargs='+', default=None)
    parser.add_argument('--word-subfolder', default='word')
    parser.add_argument('--sentence-subfolder', default='sentence')
    parser.add_argument('--include-sentences', action='store_true')
    parser.add_argument('--phrase-set-path', type=Path, default=None)
    parser.add_argument('--decode', choices=['greedy', 'beam'], default='greedy')
    parser.add_argument('--beam-width', type=int, default=10)
    parser.add_argument('--splits-json', type=Path, default=None,
                         help='optional — explicitly specify a single splits.json to use across experiments')
    parser.add_argument('--no-auto-splits', action='store_true',
                         help='skip auto-detecting splits.json next to each checkpoint')
    parser.add_argument('--out-dir', type=Path, default=Path('checkpoint_comparison'))
    args = parser.parse_args()

    has_dictionary = args.phrase_set_path is not None
    summary_rows = []
    per_experiment_dfs = {}

    for label, checkpoint_path in args.experiments:
        print(f'\n{"=" * 70}\n[COMPARE] evaluating "{label}" ({checkpoint_path})\n{"=" * 70}')
        
        splits_json = None
        if args.splits_json is not None:
            splits_json = args.splits_json
            print(f'[COMPARE] using user-specified splits.json: {splits_json}')
        elif not args.no_auto_splits:
            candidate = checkpoint_path.parent / 'splits.json'
            if candidate.exists():
                splits_json = candidate
                print(f'[COMPARE] auto-detected splits.json: {splits_json}')
            else:
                print('[COMPARE] no splits.json found next to this checkpoint — evaluating every trial found')

        eval_args = argparse.Namespace(
            checkpoint=checkpoint_path, participants_root=args.participants_root,
            participants=args.participants, word_subfolder=args.word_subfolder,
            sentence_subfolder=args.sentence_subfolder, include_sentences=args.include_sentences,
            splits_json=splits_json, phrase_set_path=args.phrase_set_path,
            decode=args.decode, beam_width=args.beam_width, quiet=True,
            out_dir=args.out_dir / label,
        )
        df = run_evaluation(eval_args)
        per_experiment_dfs[label] = df

        row = {'experiment': label, 'checkpoint': str(checkpoint_path)}
        row.update(_load_training_config(checkpoint_path))
        row.update(_summarize(df, 'word', has_dictionary))
        row.update(_summarize(df, 'sentence', has_dictionary))
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows).set_index('experiment')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out_dir / 'comparison.csv')

    print(f'\n\n{"#" * 70}\n[RESULT] comparison across {len(args.experiments)} experiments\n{"#" * 70}\n')
    with pd.option_context('display.max_columns', None, 'display.width', 200):
        print(summary.to_string())
    print(f'\n[DATA] {args.out_dir}/comparison.csv  (per-experiment per-trial CSVs also under {args.out_dir}/<label>/)')

    _plot_comparison(summary, args.out_dir, has_dictionary)


def _plot_comparison(summary: pd.DataFrame, out_dir: Path, has_dictionary: bool):
    word_acc_col = 'word_corrected_accuracy' if has_dictionary and 'word_corrected_accuracy' in summary.columns \
        else 'word_raw_accuracy'
    sentence_acc_col = 'sentence_corrected_accuracy' if has_dictionary and 'sentence_corrected_accuracy' in summary.columns \
        else 'sentence_raw_accuracy'
    word_cer_col = 'word_corrected_cer' if has_dictionary and 'word_corrected_cer' in summary.columns \
        else 'word_raw_cer'
    sentence_cer_col = 'sentence_corrected_cer' if has_dictionary and 'sentence_corrected_cer' in summary.columns \
        else 'sentence_raw_cer'

    has_sentence = sentence_acc_col in summary.columns and summary[sentence_acc_col].notna().any()
    n_cols = 2 if has_sentence else 1
    fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 8))
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    axes[0, 0].bar(summary.index, summary[word_acc_col].fillna(0), color='#1f77b4')
    axes[0, 0].set_title('Word exact-match accuracy (higher = better)')
    axes[0, 0].set_ylabel('accuracy')
    axes[0, 0].set_ylim(0, 1)
    axes[1, 0].bar(summary.index, summary[word_cer_col].fillna(0), color='#2ca02c')
    axes[1, 0].set_title('Word CER — Character Error Rate (lower = better)')
    axes[1, 0].set_ylabel('CER')

    if has_sentence:
        axes[0, 1].bar(summary.index, summary[sentence_acc_col].fillna(0), color='#ff7f0e')
        axes[0, 1].set_title('Sentence exact-match accuracy (higher = better)')
        axes[0, 1].set_ylabel('accuracy')
        axes[0, 1].set_ylim(0, 1)
        axes[1, 1].bar(summary.index, summary[sentence_cer_col].fillna(0), color='#d62728')
        axes[1, 1].set_title('Sentence CER — Character Error Rate (lower = better)')
        axes[1, 1].set_ylabel('CER')

    for ax in axes.flat:
        ax.tick_params(axis='x', rotation=30)

    fig.tight_layout()
    fig.savefig(out_dir / 'comparison.png', dpi=120)
    plt.close(fig)
    print(f'[PLOT] comparison chart saved to {out_dir}/comparison.png')


if __name__ == '__main__':
    main()