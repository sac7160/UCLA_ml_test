"""
evaluate_writeas.py
────────────────────────────────────────────────────────────────────────────
Evaluates a trained model_writeas.WriteASModel checkpoint (see
train_writeas.py) on word/sentence trials — the WriteAS counterpart to
evaluate_word_ctc.py, reusing its own CER/WER helper functions directly
so the two baselines' numbers are computed the exact same way and are
directly comparable.

Reports BOTH of WriteAS's own two outputs separately (see model_writeas.py's
docstring on why the paper has two decoders in the first place):
  - "ctc"       : the CTC head's own greedy decode — this is what the
                  paper calls the STREAMING output (Section 4.3, Fig. 1's
                  "Streaming output": "Come back soe_")
  - "attention" : the attention decoder's autoregressive decode — the
                  paper's RESCORED final result (Fig. 1's "Rescoring":
                  "Come back soon"), expected to generally be more
                  accurate since it can use full-sequence context the
                  CTC head alone cannot.

Usage:
    python evaluate_writeas.py \\
        --checkpoint checkpoints/writeas_baseline/best_model.pt \\
        --participants-root ../dataset --word-subfolder word \\
        --out-dir writeas_eval
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config_ctc
from dataset_ctc import load_imu_variable, discover_participant_dataset_dirs
from dataset_ctc_realtext import scan_text_trials, get_word_span
from model_writeas import WriteASModel
from evaluate_word_ctc import edit_distance, _cer, _wer   # reused directly — see this file's
                                                              # own top-level docstring on why


def ctc_greedy_decode(ctc_log_probs: torch.Tensor, lengths: torch.Tensor, classes: list) -> list:
    """Same collapse-repeats-then-drop-blanks logic as
    train_writeas.ctc_greedy_decode_writeas — reimplemented standalone
    here (this file intentionally doesn't import from train_writeas.py,
    keeping evaluation independent of the training script's own
    internals) rather than imported, mirroring how evaluate_word_ctc.py
    keeps its own decode logic self-contained too."""
    preds = ctc_log_probs.argmax(dim=2).transpose(0, 1)   # (B, T)
    results = []
    for i in range(preds.shape[0]):
        seq = preds[i, :lengths[i]].tolist()
        collapsed = []
        prev = None
        for s in seq:
            if s != prev:
                collapsed.append(s)
            prev = s
        letters = ''.join(classes[s - 1] for s in collapsed if s != 0)
        results.append(letters)
    return results


def attention_greedy_decode(attn_logits: torch.Tensor, classes: list, eos_idx: int) -> list:
    """Reads off the attention decoder's own argmax at each step,
    stopping at the FIRST predicted <EOS> (unlike training, which always
    runs the decoder for a fixed number of steps — see model_writeas.py's
    own docstring on this) — this is the actual inference-time behavior
    the paper's own Fig. 1 "Rescoring" output represents."""
    preds = attn_logits.argmax(dim=2)   # (B, steps)
    results = []
    for i in range(preds.shape[0]):
        letters = []
        for tok in preds[i].tolist():
            if tok == eos_idx:
                break
            if 0 <= tok < len(classes):
                letters.append(classes[tok])
        results.append(''.join(letters))
    return results


def run_evaluation(args) -> pd.DataFrame:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    classes = ckpt['classes']
    imu_source = ckpt.get('imu_source', config_ctc.DEFAULT_IMU_SOURCE)
    finger = ckpt.get('finger', config_ctc.DEFAULT_FINGER)
    n_classes = ckpt.get('n_classes', len(classes))
    print(f'[SETUP] checkpoint: classes={classes} imu_source={imu_source} device={device}')

    model = WriteASModel(n_classes=n_classes, sample_rate=config_ctc.IMU_RESAMPLE_HZ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    eos_idx = model.eos_idx

    participant_dirs = discover_participant_dataset_dirs(args.participants_root, args.participants)
    if not participant_dirs:
        raise RuntimeError(f'no participant dataset folders found under {args.participants_root}')

    test_only_dirs = None
    if args.splits_json is not None:
        with open(args.splits_json) as f:
            payload = json.load(f)
        text_splits = payload.get('text_splits')
        if text_splits is not None:
            test_only_dirs = {Path(d).resolve() for d, _ in text_splits['test']}
            print(f'[SETUP] restricting to {len(test_only_dirs)} held-out test trials in '
                  f'{args.splits_json}')

    rows = []
    for participant_name, pdir in participant_dirs:
        trials = scan_text_trials(pdir, args.word_subfolder, imu_source=imu_source)
        if args.sentence_subfolder:
            trials += scan_text_trials(pdir, args.sentence_subfolder, imu_source=imu_source)
        if test_only_dirs is not None:
            trials = [(d, t) for d, t in trials if d.resolve() in test_only_dirs]
        print(f'[SETUP]   {participant_name}: {len(trials)} trials')

        for trial_dir, true_word in trials:
            t_start, t_end = get_word_span(trial_dir)
            imu = load_imu_variable(trial_dir, imu_source, finger, t_start, t_end).unsqueeze(0).to(device)
            imu_len = torch.tensor([imu.shape[2]]).to(device)

            with torch.no_grad():
                ctc_log_probs, n_clips, attn_logits = model(
                    imu, imu_len, target_tokens=None, max_decode_len=args.max_decode_len)

            ctc_pred = ctc_greedy_decode(ctc_log_probs, n_clips, classes)[0]
            attn_pred = attention_greedy_decode(attn_logits, classes, eos_idx)[0]
            true_letters = ''.join(c for c in true_word if c in classes)

            row = {
                'participant': participant_name, 'trial': str(trial_dir), 'true_word': true_letters,
                'ctc_pred': ctc_pred, 'ctc_correct': ctc_pred == true_letters,
                'ctc_edit_distance': edit_distance(ctc_pred, true_letters),
                'attention_pred': attn_pred, 'attention_correct': attn_pred == true_letters,
                'attention_edit_distance': edit_distance(attn_pred, true_letters),
            }
            rows.append(row)
            if not args.quiet:
                mark_ctc = 'OK' if row['ctc_correct'] else '  '
                mark_attn = 'OK' if row['attention_correct'] else '  '
                print(f'  {participant_name}  "{true_letters}" -> '
                      f'ctc[{mark_ctc}]="{ctc_pred}" attn[{mark_attn}]="{attn_pred}"')

    return pd.DataFrame(rows)


def _summarize(df: pd.DataFrame, pred_col: str, correct_col: str, ed_col: str) -> dict:
    return {
        'n': int(len(df)),
        'exact_accuracy': round(float(df[correct_col].mean()), 4),
        'cer': round(float(_cer(df, pred_col, ed_col)), 4),
        'wer': round(float(_wer(df, pred_col)), 4) if not pd.isna(_wer(df, pred_col)) else None,
        'mean_edit_distance': round(float(df[ed_col].mean()), 4),
    }


def build_arg_parser():
    parser = argparse.ArgumentParser(description='Evaluates a WriteAS baseline checkpoint (see '
                                                   'train_writeas.py), reporting both its CTC and '
                                                   'attention-decoder outputs.')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--participants-root', type=Path, required=True)
    parser.add_argument('--participants', nargs='+', default=None)
    parser.add_argument('--word-subfolder', default='word')
    parser.add_argument('--sentence-subfolder', default=None)
    parser.add_argument('--splits-json', type=Path, default=None,
                         help='optional — train_writeas.py doesn\'t currently save one (it does its '
                              'own random train/val split each run rather than a saved, reusable '
                              'splits.json — see train_writeas.py\'s own module docstring). If you '
                              'have no splits.json, this evaluates every trial found, which may '
                              'include some the checkpoint was trained on.')
    parser.add_argument('--max-decode-len', type=int, default=20)
    parser.add_argument('--out-dir', type=Path, default=Path('writeas_eval'))
    parser.add_argument('--quiet', action='store_true')
    return parser


def main():
    args = build_arg_parser().parse_args()
    df = run_evaluation(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dir / 'writeas_results.csv', index=False)

    print(f'\n[RESULT] {len(df)} trials evaluated')
    report = {'checkpoint': str(args.checkpoint)}
    for decoder, pred_col, correct_col, ed_col in [
        ('ctc', 'ctc_pred', 'ctc_correct', 'ctc_edit_distance'),
        ('attention', 'attention_pred', 'attention_correct', 'attention_edit_distance'),
    ]:
        metrics = _summarize(df, pred_col, correct_col, ed_col)
        report[decoder] = metrics
        print(f'\n[RESULT] === {decoder.upper()} decoder ===')
        print(f'  accuracy={metrics["exact_accuracy"]:.3f}  cer={metrics["cer"]:.3f}  '
              f'wer={metrics["wer"]}  mean_edit_distance={metrics["mean_edit_distance"]:.2f}')

    with open(args.out_dir / 'report.json', 'w') as f:
        json.dump(report, f, indent=2)
    print(f'\n[DATA] {args.out_dir}/writeas_results.csv  (per-trial predictions, both decoders)')
    print(f'[DATA] {args.out_dir}/report.json  (accuracy/CER/WER, both decoders)')


if __name__ == '__main__':
    main()
