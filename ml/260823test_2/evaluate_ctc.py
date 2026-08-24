"""
evaluate_word_ctc.py
────────────────────────────────────────────────────────────────────────────
Runs a CTC-trained letter model (train_ctc.py) on WHOLE, un-segmented word
trials — first audio_touch_on to last audio_touch_off, exactly as
requested (internal touch_on/touch_off boundaries between individual
letters inside a word are unreliable, so this deliberately does NOT try
to pre-segment on them the way evaluate_word_via_letter_hidden_states.py
did; CTC's whole purpose is finding its own alignment between the
continuous signal and the letter sequence instead).

For each word trial:
  1. Read trial_info.json for the ground-truth word (`content`).
  2. Read events.csv, take the FIRST audio_touch_on and LAST
     audio_touch_off as the trial's effective [start, end] window —
     everything else in events.csv is ignored.
  3. Load that whole window's audio/IMU via dataset_ctc.py's variable-
     length loaders (same preprocessing train_ctc.py trained on) and run
     it through the model in one pass.
  4. CTC greedy-decode the output into a raw letter string.
  5. Snap that raw string to its nearest MacKenzie-phrase-set word (by
     edit distance) as a second, dictionary-constrained prediction —
     since the true word is always drawn from that same pool.

Reports, separately: raw CTC string accuracy (exact match, no dictionary
help at all) and dictionary-snapped accuracy, plus mean edit distance for
each — the gap between the two says how much of the remaining error is
"got the letters almost right but CTC alone can't spell-correct" versus
"the underlying letter predictions were the actual problem".

Usage:
    python evaluate_word_ctc.py \\
        --checkpoint checkpoints/letter_ctc_fingertip/best_model.pt \\
        --word-dataset-root ../dataset/word \\
        --phrase-set-path ../collector/core/phrases2.txt
"""

import argparse
import json
import sys
from pathlib import Path

# See train_ctc.py's identical block for why this is needed — config.py/
# dataset.py/model.py live one level up from this file's own subfolder.
# dataset_ctc.py (imported below) does `import config_ctc`, which itself
# does `import config` — sys.path being process-wide means fixing it up
# once here, before that import chain runs, is enough; no need to repeat
# it in dataset_ctc.py/config_ctc.py themselves.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import torch

from dataset_ctc import load_audio_variable, load_imu_variable, discover_participant_dataset_dirs
from model_ctc import build_model_ctc, ctc_greedy_decode, ctc_beam_search_decode
from autocorrect_ctc import CharBigramLM, autocorrect


def get_word_span(trial_dir: Path):
    """Returns (first_touch_on, last_touch_off) in trial-relative seconds,
    or None if events.csv has no complete on/off pair. Deliberately
    ignores every touch event in between — see module docstring."""
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
    parser = argparse.ArgumentParser(description='Decode words by running a CTC letter model on '
                                                  'whole, un-segmented word trials')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--participants-root', type=Path,
                         default=Path(__file__).resolve().parent.parent.parent / 'dataset',
                         help='folder containing one subfolder per participant (p1/, p2/, p3/, ...), each '
                              'with its own dataset/word/trial_XXX/ inside — every "p*"-named folder found '
                              'here is evaluated automatically; adding a new participant folder later needs '
                              'no command-line change. See train_ctc.py\'s identical argument for the exact '
                              'expected layout.')
    parser.add_argument('--participants', nargs='+', default=None,
                         help='restrict to specific participants (e.g. --participants p1 p3) — default: '
                              'use every participant folder found under --participants-root')
    parser.add_argument('--word-subfolder', default='word',
                         help='which subfolder under each participant\'s dataset/ holds word trials '
                              '(default "word", matching dataset/<participant>/dataset/word/trial_XXX/)')
    parser.add_argument('--sentence-subfolder', default='sentence',
                         help='which subfolder under each participant\'s dataset/ holds sentence trials — '
                              'only scanned when --include-sentences is set, or when --splits-json restricts '
                              'to a test split that happens to include sentence trials (train_ctc.py\'s '
                              '--use-sentence-trials pools word and sentence trials together before '
                              'splitting, so a given test split can contain either regardless of this flag).')
    parser.add_argument('--include-sentences', action='store_true',
                         help='also evaluate sentence trials (dataset/<participant>/dataset/sentence/'
                              'trial_XXX/), alongside the word trials this script always evaluates. Reported '
                              'SEPARATELY from word results (see the per-writing-target [RESULT] blocks) — '
                              'sentences are longer and structurally harder to decode, so averaging them '
                              'together with words would make neither number meaningful.')
    parser.add_argument('--splits-json', type=Path, default=None,
                         help='optional — a checkpoint\'s splits.json (see train_ctc.py). If it contains a '
                              '"text_splits" key (i.e. train_ctc.py was run with --use-word-trials/'
                              '--use-sentence-trials), evaluation is restricted to that split\'s "test" '
                              'trials only — otherwise trials the model was directly trained on would '
                              'inflate the accuracy reported here. Omit this if the checkpoint was trained '
                              'WITHOUT --use-word-trials/--use-sentence-trials — every word trial found is '
                              'genuinely held-out in that case, since none were used for training at all.')
    parser.add_argument('--phrase-set-path', type=Path, default=None,
                         help='optional — enables a second, dictionary-corrected accuracy metric on top of '
                              'the raw CTC decode (see snap_to_dictionary()\'s docstring). The ground-truth '
                              'word always comes straight from trial_info.json\'s `content` field regardless '
                              'of whether this is set — this only affects how the model\'s OWN prediction '
                              'gets scored. Omit it to skip that step and see raw, uncorrected CTC accuracy '
                              'only.')
    parser.add_argument('--out-dir', type=Path, default=Path('word_ctc_eval'))
    parser.add_argument('--decode', choices=['greedy', 'beam'], default='greedy',
                         help='CTC decoding strategy for raw_ctc_pred — "greedy" (default) commits to the '
                              'single highest-probability symbol at every timestep and can never recover '
                              'from an early mistake; "beam" (CTC prefix beam search — see model_ctc.py\'s '
                              'ctc_beam_search_decode) keeps --beam-width alternative hypotheses alive '
                              'throughout decoding, which tends to matter most for longer trials '
                              '(sentences especially) where one wrong argmax early on derails everything '
                              'after it under greedy decode. Beam search is slower per-trial than greedy — '
                              'noticeable on long sentence trials — but never changes what happens during '
                              'training, only how a trained model\'s output gets decoded at evaluation time.')
    parser.add_argument('--beam-width', type=int, default=10,
                         help='number of alternative hypotheses kept alive at each timestep when '
                              '--decode beam is set (ignored for --decode greedy). Larger values search '
                              'more thoroughly at the cost of proportionally slower decoding; beam_width=1 '
                              'is mathematically equivalent to (a slightly slower) greedy decode.')
    parser.add_argument('--quiet', action='store_true',
                         help='suppress the per-trial (true -> predicted) lines printed by default — '
                              'only the aggregate [RESULT] summary is shown')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    classes = ckpt['classes']
    modality = ckpt.get('modality', 'imu')
    rnn_hidden = ckpt.get('rnn_hidden', 256)   
    use_space = ckpt.get('use_space', True)    
    sequence_encoder = ckpt.get('sequence_encoder', 'gru')   
    transformer_layers = ckpt.get('transformer_layers', 2)
    transformer_nhead = ckpt.get('transformer_nhead', 8)
    audio_source = ckpt.get('audio_source', 'surface')
    imu_source = ckpt.get('imu_source', 'fingertip')
    finger = ckpt.get('finger', 'index')
    print(f'[SETUP] checkpoint: modality={modality} audio_source={audio_source} '
          f'imu_source={imu_source} classes={classes} use_space={use_space} '
          f'sequence_encoder={sequence_encoder}')
    decode_desc = f'beam search (width={args.beam_width})' if args.decode == 'beam' else 'greedy'
    print(f'[SETUP] decoding: {decode_desc}')

    model = build_model_ctc(modality, n_classes=len(classes), rnn_hidden=rnn_hidden,
                             use_space=use_space, sequence_encoder=sequence_encoder,
                             transformer_layers=transformer_layers,
                             transformer_nhead=transformer_nhead).to(device)
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
        print(f'[SETUP] dictionary: {len(word_dictionary)} words, {len(sentence_dictionary)} sentences '
              f'(context-aware autocorrected accuracy will be computed)')
    else:
        print('[SETUP] no --phrase-set-path given — reporting raw CTC accuracy only')

    test_only_dirs = None   
    if args.splits_json is not None:
        with open(args.splits_json) as f:
            splits_payload = json.load(f)
        text_splits = splits_payload.get('text_splits')
        if text_splits is None:
            print(f'[SETUP] {args.splits_json} has no "text_splits" — evaluating every word trial found.')
        else:
            test_only_dirs = {Path(trial_dir).resolve() for trial_dir, _ in text_splits['test']}
            print(f'[SETUP] restricting evaluation to the {len(test_only_dirs)} held-out test trials in '
                  f'{args.splits_json}')

    participant_dirs = discover_participant_dataset_dirs(args.participants_root, args.participants)
    if not participant_dirs:
        raise RuntimeError(f'no participant dataset folders found under {args.participants_root}')
    print(f'[SETUP] participants: {[name for name, _ in participant_dirs]}')

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
    """Prints comprehensive evaluation metrics including CER and per-participant breakdowns."""
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