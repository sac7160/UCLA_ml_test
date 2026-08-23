"""
evaluate_word_ctc.py (for GRU-based LetterCTCNet)
────────────────────────────────────────────────────────────────────────────
Runs a CTC-trained letter model (train_ctc.py, GRU version) on WHOLE,
un-segmented word/sentence trials — first audio_touch_on to last audio_touch_off.

Reports, separately for word and sentence trials:
  - Trial Count (n)
  - Raw Exact-Match Accuracy
  - Raw Mean Edit Distance
  - Character Error Rate (CER)
  - Blank-only Predictions (%)
  - Dictionary-Corrected Accuracy & CER (if --phrase-set-path is provided)
  - Per-Participant Breakdown (Raw Accuracy, Mean Edit Distance, CER)
"""

import argparse
import json
import sys
from pathlib import Path

# Fix sys.path to access parent folder modules (config_ctc, dataset_ctc, model_ctc, etc.)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import torch

from dataset_ctc import load_audio_variable, load_imu_variable, discover_participant_dataset_dirs
from model_ctc import build_model_ctc, ctc_greedy_decode, ctc_beam_search_decode
from autocorrect_ctc import CharBigramLM, autocorrect


def get_word_span(trial_dir: Path):
    """Returns (first_touch_on, last_touch_off) in trial-relative seconds,
    or None if events.csv has no complete on/off pair."""
    events_path = trial_dir / 'events.csv'
    if not events_path.exists():
        return None
    df = pd.read_csv(events_path).sort_values('time_aligned')
    on_times = df[df['event'] == 'audio_touch_on']['time_aligned']
    off_times = df[df['event'] == 'audio_touch_off']['time_aligned']
    if on_times.empty or off_times.empty:
        return None
    return float(on_times.iloc[0]), float(off_times.iloc[-1])


def edit_distance(a: str, b: str) -> int:
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, dp[0] = dp[0], i
        for j, cb in enumerate(b, 1):
            prev, dp[j] = dp[j], min(dp[j] + 1, dp[j - 1] + 1, prev + (ca != cb))
    return dp[-1]


def calculate_cer(df) -> float:
    """Calculates Character Error Rate (CER): total edit distance / total true character length."""
    total_edits = df['raw_edit_distance'].sum()
    total_chars = df['true_word'].str.len().sum()
    if total_chars == 0:
        return 0.0
    return float(total_edits / total_chars)


def main():
    parser = argparse.ArgumentParser(description='Evaluate GRU-based CTC letter model on whole trials')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--participants-root', type=Path,
                         default=Path(__file__).resolve().parent.parent.parent / 'dataset',
                         help='folder containing participant subfolders (p1/, p2/, ...)')
    parser.add_argument('--participants', nargs='+', default=None,
                         help='restrict to specific participants (e.g. --participants p1 p3)')
    parser.add_argument('--word-subfolder', default='word',
                         help='subfolder for word trials (default "word")')
    parser.add_argument('--sentence-subfolder', default='sentence',
                         help='subfolder for sentence trials (default "sentence")')
    parser.add_argument('--include-sentences', action='store_true',
                         help='also evaluate sentence trials alongside word trials')
    parser.add_argument('--splits-json', type=Path, default=None,
                         help='optional — checkpoint\'s splits.json to restrict evaluation to test split only')
    parser.add_argument('--phrase-set-path', type=Path, default=None,
                         help='optional — path to phrases2.txt for dictionary-corrected accuracy/CER')
    parser.add_argument('--out-dir', type=Path, default=Path('word_ctc_eval'))
    parser.add_argument('--decode', choices=['greedy', 'beam'], default='greedy',
                         help='CTC decoding strategy ("greedy" or "beam")')
    parser.add_argument('--beam-width', type=int, default=10,
                         help='beam width for beam search decoding')
    parser.add_argument('--quiet', action='store_true',
                         help='suppress per-trial logging')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    classes = ckpt['classes']
    modality = ckpt.get('modality', 'imu')
    rnn_hidden = ckpt.get('rnn_hidden', 256)
    use_space = ckpt.get('use_space', True)
    audio_source = ckpt.get('audio_source', 'surface')
    imu_source = ckpt.get('imu_source', 'fingertip')
    finger = ckpt.get('finger', 'index')

    print(f'[SETUP] checkpoint (GRU): modality={modality} audio_source={audio_source} '
          f'imu_source={imu_source} classes={classes} use_space={use_space} rnn_hidden={rnn_hidden}')
    decode_desc = f'beam search (width={args.beam_width})' if args.decode == 'beam' else 'greedy'
    print(f'[SETUP] decoding: {decode_desc}')

    # GRU 모델 빌드 (transformer 인자 제거됨)
    model = build_model_ctc(modality, n_classes=len(classes), rnn_hidden=rnn_hidden,
                             use_space=use_space).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    word_dictionary = None
    sentence_dictionary = None
    word_lm = None
    sentence_lm = None
    if args.phrase_set_path is not None:
        import phrase_set
        phrase_set._load(args.phrase_set_path)
        word_dictionary = list(phrase_set._words)
        sentence_dictionary = list(phrase_set._phrases)
        word_lm = CharBigramLM(word_dictionary)
        sentence_lm = CharBigramLM(sentence_dictionary)
        print(f'[SETUP] dictionary: {len(word_dictionary)} words, {len(sentence_dictionary)} sentences loaded.')
    else:
        print('[SETUP] no --phrase-set-path given — reporting raw CTC accuracy only.')

    test_only_dirs = None
    if args.splits_json is not None:
        with open(args.splits_json) as f:
            splits_payload = json.load(f)
        text_splits = splits_payload.get('text_splits')
        if text_splits is not None:
            test_only_dirs = {Path(trial_dir).resolve() for trial_dir, _ in text_splits['test']}
            print(f'[SETUP] restricting evaluation to {len(test_only_dirs)} test trials from {args.splits_json}')

    participant_dirs = discover_participant_dataset_dirs(args.participants_root, args.participants)
    if not participant_dirs:
        raise RuntimeError(f'no participant dataset folders found under {args.participants_root}')

    rows = []
    scan_sentences = args.include_sentences or (test_only_dirs is not None)
    with torch.no_grad():
        for participant_name, pdir in participant_dirs:
            word_dir = pdir / args.word_subfolder
            trial_dirs = sorted(p for p in word_dir.glob('trial_*') if p.is_dir()) if word_dir.exists() else []
            n_word = len(trial_dirs)
            n_sentence = 0
            if scan_sentences:
                sentence_dir = pdir / args.sentence_subfolder
                if sentence_dir.exists():
                    sentence_trial_dirs = sorted(p for p in sentence_dir.glob('trial_*') if p.is_dir())
                    n_sentence = len(sentence_trial_dirs)
                    trial_dirs += sentence_trial_dirs
            if test_only_dirs is not None:
                trial_dirs = [t for t in trial_dirs if t.resolve() in test_only_dirs]
            
            print(f'[SETUP]   {participant_name}: {n_word} word trials'
                  + (f', {n_sentence} sentence trials' if scan_sentences else ''))

            for trial_dir in trial_dirs:
                info_path = trial_dir / 'trial_info.json'
                if not info_path.exists():
                    continue
                with open(info_path) as f:
                    info = json.load(f)
                true_word = info.get('content', '').strip().lower()
                if not use_space:
                    true_word = true_word.replace(' ', '')
                if not true_word:
                    continue

                span = get_word_span(trial_dir)
                if span is None:
                    continue
                t_start, t_end = span

                audio = None
                imu = None
                if modality in ('audio', 'fusion'):
                    audio = load_audio_variable(trial_dir, audio_source, t_start, t_end).unsqueeze(0).to(device)
                if modality in ('imu', 'fusion'):
                    imu = load_imu_variable(trial_dir, imu_source, finger, t_start, t_end).unsqueeze(0).to(device)
                audio_len = torch.tensor([audio.shape[3]]).to(device) if audio is not None else None
                imu_len = torch.tensor([imu.shape[2]]).to(device) if imu is not None else None

                log_probs, out_len, _traj_pred, _spec_recon = model(audio, audio_len, imu, imu_len)
                if args.decode == 'beam':
                    raw_pred = ctc_beam_search_decode(log_probs, out_len, classes,
                                                       beam_width=args.beam_width, use_space=use_space)[0]
                else:
                    raw_pred = ctc_greedy_decode(log_probs, out_len, classes, use_space=use_space)[0]
                
                raw_ed = edit_distance(raw_pred, true_word)
                is_sentence = trial_dir.parent.name == args.sentence_subfolder

                row = {
                    'participant': participant_name, 'writing_target': 'sentence' if is_sentence else 'word',
                    'trial': str(trial_dir), 'true_word': true_word,
                    'span_sec': round(t_end - t_start, 3),
                    'raw_ctc_pred': raw_pred, 'raw_correct': raw_pred == true_word,
                    'raw_edit_distance': raw_ed,
                }
                if word_dictionary is not None:
                    candidates = sentence_dictionary if is_sentence else word_dictionary
                    lm = sentence_lm if is_sentence else word_lm
                    corrected_pred = autocorrect(raw_pred, candidates, lm)
                    row.update({
                        'dict_snapped_pred': corrected_pred, 'dict_correct': corrected_pred == true_word,
                        'dict_edit_distance': edit_distance(corrected_pred, true_word),
                    })
                rows.append(row)

                if not args.quiet:
                    mark = 'OK  ' if raw_pred == true_word else '    '
                    pred_shown = raw_pred if raw_pred else '(blank)'
                    tag = 'S' if is_sentence else 'W'
                    line = (f'  [{mark}][{tag}] {participant_name}  "{true_word}"({len(true_word)}) -> '
                            f'"{pred_shown}"({len(raw_pred)})  edit={raw_ed}  span={t_end - t_start:.2f}s')
                    if word_dictionary is not None:
                        line += f'  | corrected->"{corrected_pred}"' + (' OK' if corrected_pred == true_word else '')
                    print(line)

    df = pd.DataFrame(rows)
    if len(df) == 0:
        print('[WARNING] No valid trials evaluated.')
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dir / 'word_ctc_results.csv', index=False)

    print(f'\n[RESULT] {len(df)} trials evaluated across {len(participant_dirs)} participants '
          f'({(df["writing_target"] == "word").sum()} word, {(df["writing_target"] == "sentence").sum()} sentence)')
    
    for target in ('word', 'sentence'):
        sub = df[df['writing_target'] == target]
        if len(sub) == 0:
            continue
        print(f'\n[RESULT] === {target.upper()} trials ({len(sub)}) ===')
        _print_summary_block(sub, has_dictionary=word_dictionary is not None)
    print(f'\n[DATA] {args.out_dir}/word_ctc_results.csv')


def _print_summary_block(df, has_dictionary: bool):
    """Prints comprehensive evaluation metrics including CER and per-participant breakdown."""
    n_trials = len(df)
    n_blank = (df['raw_ctc_pred'] == '').sum()
    true_lens = df['true_word'].str.len()
    pred_lens = df['raw_ctc_pred'].str.len()
    
    raw_acc = df['raw_correct'].mean()
    raw_mean_ed = df['raw_edit_distance'].mean()
    cer = calculate_cer(df)
    blank_pct = (n_blank / n_trials) * 100 if n_trials > 0 else 0.0

    print(f'[RESULT] Trial Count (n):                 {n_trials}')
    print(f'[RESULT] Raw Exact-Match Accuracy:       {raw_acc:.3f} ({raw_acc * 100:.1f}%)')
    print(f'[RESULT] Raw Mean Edit Distance:         {raw_mean_ed:.2f} characters')
    print(f'[RESULT] Character Error Rate (CER):     {cer:.3f} ({cer * 100:.1f}%)')
    print(f'[RESULT] Blank-only Predictions:         {n_blank}/{n_trials} ({blank_pct:.1f}%)')
    print(f'[RESULT] True Text Length:               mean={true_lens.mean():.1f} (min={true_lens.min()}, max={true_lens.max()})')
    print(f'[RESULT] Predicted Length:               mean={pred_lens.mean():.1f} (min={pred_lens.min()}, max={pred_lens.max()})')

    if pred_lens.mean() < true_lens.mean() * 0.5:
        print('[NOTE]   Predictions are much SHORTER than true text — model shows blank-collapse tendency.')

    if has_dictionary:
        dict_acc = df['dict_correct'].mean()
        dict_mean_ed = df['dict_edit_distance'].mean()
        dict_cer = df['dict_edit_distance'].sum() / df['true_word'].str.len().sum() if df['true_word'].str.len().sum() > 0 else 0.0
        print(f'[RESULT] Dictionary-Corrected Accuracy:  {dict_acc:.3f} ({dict_acc * 100:.1f}%)')
        print(f'[RESULT] Dictionary Corrected Mean Edit: {dict_mean_ed:.2f} characters')
        print(f'[RESULT] Dictionary Corrected CER:       {dict_cer:.3f} ({dict_cer * 100:.1f}%)')

    print('\n[RESULT] --- Per-Participant Breakdown ---')
    participant_stats = []
    for participant, group in df.groupby('participant'):
        p_trials = len(group)
        p_raw_acc = group['raw_correct'].mean()
        p_raw_ed = group['raw_edit_distance'].mean()
        p_cer = calculate_cer(group)
        p_stat = {
            'participant': participant,
            'trials': p_trials,
            'raw_acc': round(p_raw_acc, 3),
            'raw_mean_ed': round(p_raw_ed, 2),
            'cer': round(p_cer, 3)
        }
        if has_dictionary:
            p_dict_acc = group['dict_correct'].mean()
            p_stat['dict_acc'] = round(p_dict_acc, 3)
        participant_stats.append(p_stat)

    p_df = pd.DataFrame(participant_stats)
    print(p_df.to_string(index=False))


if __name__ == '__main__':
    main()