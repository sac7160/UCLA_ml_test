# """
# evaluate_word_ctc.py
# ────────────────────────────────────────────────────────────────────────────
# Runs a CTC-trained letter model (train_ctc.py) on WHOLE, un-segmented word
# trials — first audio_touch_on to last audio_touch_off, exactly as
# requested (internal touch_on/touch_off boundaries between individual
# letters inside a word are unreliable, so this deliberately does NOT try
# to pre-segment on them the way evaluate_word_via_letter_hidden_states.py
# did; CTC's whole purpose is finding its own alignment between the
# continuous signal and the letter sequence instead).

# For each word trial:
#   1. Read trial_info.json for the ground-truth word (`content`).
#   2. Read events.csv, take the FIRST audio_touch_on and LAST
#      audio_touch_off as the trial's effective [start, end] window —
#      everything else in events.csv is ignored.
#   3. Load that whole window's audio/IMU via dataset_ctc.py's variable-
#      length loaders (same preprocessing train_ctc.py trained on) and run
#      it through the model in one pass.
#   4. CTC greedy-decode the output into a raw letter string.
#   5. Snap that raw string to its nearest MacKenzie-phrase-set word (by
#      edit distance) as a second, dictionary-constrained prediction —
#      since the true word is always drawn from that same pool.

# Reports, separately: raw CTC string accuracy (exact match, no dictionary
# help at all) and dictionary-snapped accuracy, plus mean edit distance for
# each — the gap between the two says how much of the remaining error is
# "got the letters almost right but CTC alone can't spell-correct" versus
# "the underlying letter predictions were the actual problem".

# Usage:
#     python evaluate_word_ctc.py \\
#         --checkpoint checkpoints/letter_ctc_fingertip/best_model.pt \\
#         --word-dataset-root ../dataset/word \\
#         --phrase-set-path ../collector/core/phrases2.txt
# """

# import argparse
# import json
# import sys
# from pathlib import Path

# # See train_ctc.py's identical block for why this is needed — config.py/
# # dataset.py/model.py live one level up from this file's own subfolder.
# # dataset_ctc.py (imported below) does `import config_ctc`, which itself
# # does `import config` — sys.path being process-wide means fixing it up
# # once here, before that import chain runs, is enough; no need to repeat
# # it in dataset_ctc.py/config_ctc.py themselves.
# sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# import pandas as pd
# import torch

# from dataset_ctc import load_audio_variable, load_imu_variable, discover_participant_dataset_dirs
# from model_ctc import build_model_ctc, ctc_greedy_decode, ctc_beam_search_decode
# from autocorrect_ctc import CharBigramLM, autocorrect


# def get_word_span(trial_dir: Path):
#     """Returns (first_touch_on, last_touch_off) in trial-relative seconds,
#     or None if events.csv has no complete on/off pair. Deliberately
#     ignores every touch event in between — see module docstring."""
#     events_path = trial_dir / 'events.csv'
#     if not events_path.exists():
#         return None
#     df = pd.read_csv(events_path).sort_values('time_aligned')
#     on_times = df[df['event'] == 'audio_touch_on']['time_aligned']
#     off_times = df[df['event'] == 'audio_touch_off']['time_aligned']
#     if on_times.empty or off_times.empty:
#         return None
#     return float(on_times.iloc[0]), float(off_times.iloc[-1])


# def edit_distance(a: str, b: str) -> int:
#     dp = list(range(len(b) + 1))
#     for i, ca in enumerate(a, 1):
#         prev, dp[0] = dp[0], i
#         for j, cb in enumerate(b, 1):
#             prev, dp[j] = dp[j], min(dp[j] + 1, dp[j - 1] + 1, prev + (ca != cb))
#     return dp[-1]


# def _word_edit_distance(a_words: list, b_words: list) -> int:
#     """Same Levenshtein recurrence as edit_distance() above, just
#     operating on a list of WORDS instead of a string of CHARACTERS —
#     used for WER below."""
#     dp = list(range(len(b_words) + 1))
#     for i, wa in enumerate(a_words, 1):
#         prev, dp[0] = dp[0], i
#         for j, wb in enumerate(b_words, 1):
#             prev, dp[j] = dp[j], min(dp[j] + 1, dp[j - 1] + 1, prev + (wa != wb))
#     return dp[-1]


# def _cer(df, pred_col: str, ed_col: str) -> float:
#     """Character Error Rate, MICRO-averaged: total edit distance summed
#     across every trial, divided by the total number of ground-truth
#     characters summed across every trial — NOT the mean of each trial's
#     own (edit_distance / length) ratio, and NOT the same thing as mean
#     edit distance (which weighs a 3-letter word's edit=1 the same as an
#     11-letter word's edit=1, even though the relative error is very
#     different). This is the standard metric OCR/ASR/handwriting-
#     recognition papers report for exactly the reason exact-match
#     accuracy alone doesn't capture: two predictions that are both
#     "wrong" by exact-match can be very different in how close they
#     actually got, and CER is what quantifies that — e.g. a 21-character
#     sentence off by only 5 characters scores 0 on exact-match but
#     ~0.24 CER, which is the more informative number for judging how
#     good the model's decode actually was."""
#     total_chars = df['true_word'].str.len().sum()
#     if total_chars == 0:
#         return float('nan')
#     return df[ed_col].sum() / total_chars


# def _wer(df, pred_col: str) -> float:
#     """Word Error Rate, same micro-averaging idea as CER but at word
#     granularity (split on whitespace) instead of character granularity —
#     complements CER: a model can have a low CER (mostly right letters)
#     while still getting very few WHOLE words exactly right if its
#     mistakes are spread across many different words, which CER alone
#     doesn't distinguish. Only meaningful when the ground truth actually
#     contains spaces — returns NaN otherwise (word trials never have
#     spaces in their content, and a --no-space checkpoint's sentence rows
#     have their spaces stripped before this DataFrame is ever built — see
#     run_evaluation's use_space handling)."""
#     if not df['true_word'].str.contains(' ').any():
#         return float('nan')
#     total_words = 0
#     total_word_edits = 0
#     for true_text, pred_text in zip(df['true_word'], df[pred_col]):
#         true_words = true_text.split()
#         pred_words = pred_text.split()
#         total_words += len(true_words)
#         total_word_edits += _word_edit_distance(true_words, pred_words)
#     return total_word_edits / total_words if total_words else float('nan')


# def _summarize_metrics(df: pd.DataFrame, target: str, has_dictionary: bool) -> dict:
#     """Flattens one writing_target's ('letter'/'word'/'sentence') slice
#     of the results into a single dict of the metrics a paper would
#     actually report: exact-match accuracy, CER, and (where meaningful —
#     see _wer's own docstring on why this is NaN for letters/most words)
#     WER, raw and dictionary-corrected. This is what report.json's
#     per-target blocks are built from — see main()."""
#     sub = df[df['writing_target'] == target]
#     if len(sub) == 0:
#         return {}
#     out = {
#         'n': int(len(sub)),
#         'raw_accuracy': round(float(sub['raw_correct'].mean()), 4),
#         'raw_cer': round(float(_cer(sub, 'raw_ctc_pred', 'raw_edit_distance')), 4),
#         'raw_wer': round(float(_wer(sub, 'raw_ctc_pred')), 4) if not pd.isna(_wer(sub, 'raw_ctc_pred')) else None,
#         'raw_mean_edit_distance': round(float(sub['raw_edit_distance'].mean()), 4),
#     }
#     if has_dictionary and 'dict_correct' in sub.columns:
#         wer_corrected = _wer(sub, 'dict_snapped_pred')
#         out.update({
#             'corrected_accuracy': round(float(sub['dict_correct'].mean()), 4),
#             'corrected_cer': round(float(_cer(sub, 'dict_snapped_pred', 'dict_edit_distance')), 4),
#             'corrected_wer': round(float(wer_corrected), 4) if not pd.isna(wer_corrected) else None,
#             'corrected_mean_edit_distance': round(float(sub['dict_edit_distance'].mean()), 4),
#         })
#     return out


# def build_arg_parser():
#     parser = argparse.ArgumentParser(description='Decode words by running a CTC letter model on '
#                                                   'whole, un-segmented word trials')
#     parser.add_argument('--checkpoint', type=Path, required=True)
#     parser.add_argument('--participants-root', type=Path,
#                          default=Path(__file__).resolve().parent.parent.parent / 'dataset',
#                          help='folder containing one subfolder per participant (p1/, p2/, p3/, ...), each '
#                               'with its own dataset/word/trial_XXX/ inside — every "p*"-named folder found '
#                               'here is evaluated automatically; adding a new participant folder later needs '
#                               'no command-line change. See train_ctc.py\'s identical argument for the exact '
#                               'expected layout.')
#     parser.add_argument('--participants', nargs='+', default=None,
#                          help='restrict to specific participants (e.g. --participants p1 p3) — default: '
#                               'use every participant folder found under --participants-root')
#     parser.add_argument('--word-subfolder', default='word',
#                          help='which subfolder under each participant\'s dataset/ holds word trials '
#                               '(default "word", matching dataset/<participant>/dataset/word/trial_XXX/)')
#     parser.add_argument('--sentence-subfolder', default='sentence',
#                          help='which subfolder under each participant\'s dataset/ holds sentence trials — '
#                               'only scanned when --include-sentences is set, or when --splits-json restricts '
#                               'to a test split that happens to include sentence trials (train_ctc.py\'s '
#                               '--use-sentence-trials pools word and sentence trials together before '
#                               'splitting, so a given test split can contain either regardless of this flag).')
#     parser.add_argument('--include-sentences', action='store_true',
#                          help='also evaluate sentence trials (dataset/<participant>/dataset/sentence/'
#                               'trial_XXX/), alongside the word trials this script always evaluates. Reported '
#                               'SEPARATELY from word results (see the per-writing-target [RESULT] blocks) — '
#                               'sentences are longer and structurally harder to decode, so averaging them '
#                               'together with words would make neither number meaningful.')
#     parser.add_argument('--include-letters', action='store_true',
#                          help='also evaluate individual LETTER trials (dataset/<participant>/dataset/<class>/'
#                               'trial_XXX/ - the same single-letter data train_ctc.py\'s LetterDatasetCTC '
#                               'trains on), reported as its own separate [RESULT] block. When --splits-json '
#                               'is given, restricted to that checkpoint\'s held-out letter test split the '
#                               'same way word/sentence trials are restricted to the text_splits test set.')
#     parser.add_argument('--splits-json', type=Path, default=None,
#                          help='optional — a checkpoint\'s splits.json (see train_ctc.py). If it contains a '
#                               '"text_splits" key (i.e. train_ctc.py was run with --use-word-trials/'
#                               '--use-sentence-trials), evaluation is restricted to that split\'s "test" '
#                               'trials only — otherwise trials the model was directly trained on would '
#                               'inflate the accuracy reported here. Omit this if the checkpoint was trained '
#                               'WITHOUT --use-word-trials/--use-sentence-trials — every word trial found is '
#                               'genuinely held-out in that case, since none were used for training at all.')
#     parser.add_argument('--phrase-set-path', type=Path, default=None,
#                          help='optional — enables a second, dictionary-corrected accuracy metric on top of '
#                               'the raw CTC decode (see snap_to_dictionary()\'s docstring). The ground-truth '
#                               'word always comes straight from trial_info.json\'s `content` field regardless '
#                               'of whether this is set — this only affects how the model\'s OWN prediction '
#                               'gets scored. Omit it to skip that step and see raw, uncorrected CTC accuracy '
#                               'only.')
#     parser.add_argument('--out-dir', type=Path, default=Path('word_ctc_eval'))
#     parser.add_argument('--decode', choices=['greedy', 'beam'], default='greedy',
#                          help='CTC decoding strategy for raw_ctc_pred — "greedy" (default) commits to the '
#                               'single highest-probability symbol at every timestep and can never recover '
#                               'from an early mistake; "beam" (CTC prefix beam search — see model_ctc.py\'s '
#                               'ctc_beam_search_decode) keeps --beam-width alternative hypotheses alive '
#                               'throughout decoding, which tends to matter most for longer trials '
#                               '(sentences especially) where one wrong argmax early on derails everything '
#                               'after it under greedy decode. Beam search is slower per-trial than greedy — '
#                               'noticeable on long sentence trials — but never changes what happens during '
#                               'training, only how a trained model\'s output gets decoded at evaluation time.')
#     parser.add_argument('--beam-width', type=int, default=10,
#                          help='number of alternative hypotheses kept alive at each timestep when '
#                               '--decode beam is set (ignored for --decode greedy). Larger values search '
#                               'more thoroughly at the cost of proportionally slower decoding; beam_width=1 '
#                               'is mathematically equivalent to (a slightly slower) greedy decode.')
#     parser.add_argument('--quiet', action='store_true',
#                          help='suppress the per-trial (true -> predicted) lines printed by default — '
#                               'only the aggregate [RESULT] summary is shown')
#     return parser


# def run_evaluation(args) -> pd.DataFrame:
#     """Does the actual work — load checkpoint, scan trials, decode,
#     score — and returns the raw per-trial results as a DataFrame (same
#     rows that get saved to word_ctc_results.csv). Factored out of main()
#     specifically so compare_checkpoints.py can call this directly for
#     several checkpoints in a row and build a side-by-side comparison
#     table, without needing to re-implement or shell out to this
#     evaluation logic. main() below is now a thin wrapper: call this, then
#     print/save exactly what it already did."""
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
#     classes = ckpt['classes']
#     modality = ckpt.get('modality', 'imu')
#     rnn_hidden = ckpt.get('rnn_hidden', 256)   # older checkpoints predating the TCN/U-Net rewrite
#                                                  # don't have this key — 256 matches train_ctc.py's own
#                                                  # current default, so this only matters for checkpoints
#                                                  # trained with a non-default --rnn-hidden
#     use_space = ckpt.get('use_space', True)    # older checkpoints predating the space class always
#                                                  # had one implicitly — True matches that
#     sequence_encoder = ckpt.get('sequence_encoder', 'gru')   # older checkpoints predating this option
#                                                                 # were always GRU
#     transformer_layers = ckpt.get('transformer_layers', 2)
#     transformer_nhead = ckpt.get('transformer_nhead', 8)
#     motion_target = ckpt.get('motion_target', 'fingertip_imu')   # older checkpoints predating
#                                                                      # --motion-target were always this
#                                                                      # (the IMU decoder's output width —
#                                                                      # 6 vs 2 channels — depends on it, so
#                                                                      # this MUST match how the checkpoint
#                                                                      # was actually trained or
#                                                                      # load_state_dict below fails on a
#                                                                      # shape mismatch)
#     audio_source = ckpt.get('audio_source', 'surface')
#     imu_source = ckpt.get('imu_source', 'fingertip')
#     finger = ckpt.get('finger', 'index')
#     print(f'[SETUP] checkpoint: modality={modality} audio_source={audio_source} '
#           f'imu_source={imu_source} classes={classes} use_space={use_space} '
#           f'sequence_encoder={sequence_encoder} motion_target={motion_target}')
#     decode_desc = f'beam search (width={args.beam_width})' if args.decode == 'beam' else 'greedy'
#     print(f'[SETUP] decoding: {decode_desc}')

#     model = build_model_ctc(modality, n_classes=len(classes), rnn_hidden=rnn_hidden,
#                              use_space=use_space, sequence_encoder=sequence_encoder,
#                              transformer_layers=transformer_layers,
#                              transformer_nhead=transformer_nhead,
#                              motion_target=motion_target).to(device)
#     model.load_state_dict(ckpt['model_state_dict'])
#     model.eval()

#     word_dictionary = None
#     sentence_dictionary = None
#     word_lm = None
#     sentence_lm = None
#     if args.phrase_set_path is not None:
#         import phrase_set   # only needs to exist/import cleanly when actually used — see this
#                              # module's docstring for why --phrase-set-path is optional at all
#         phrase_set._load(args.phrase_set_path)
#         word_dictionary = list(phrase_set._words)
#         sentence_dictionary = list(phrase_set._phrases)
#         # Separate bigram LMs for words vs. sentences — a sentence's own
#         # character statistics (includes spaces, longer strings) are
#         # different enough from a single word's that mixing them would
#         # blur both. See autocorrect_ctc.py's module docstring for why
#         # matching a raw sentence decode against the pool of REAL
#         # sentences (rather than word-by-word) is how this captures
#         # "neighboring word" context without a separate word-segmentation
#         # step.
#         word_lm = CharBigramLM(word_dictionary)
#         sentence_lm = CharBigramLM(sentence_dictionary)
#         print(f'[SETUP] dictionary: {len(word_dictionary)} words, {len(sentence_dictionary)} sentences '
#               f'(context-aware autocorrected accuracy will be computed)')
#     else:
#         print('[SETUP] no --phrase-set-path given — reporting raw CTC accuracy only')

#     test_only_dirs = None   # None = evaluate every word trial found; a set = restrict to just these
#     letter_test_only_dirs = None   # same idea, for --include-letters — a SEPARATE set, since letter
#                                      # trials are split via save_splits()'s own "splits" key, not
#                                      # "text_splits" (see dataset.py's save_splits/make_splits)
#     if args.splits_json is not None:
#         with open(args.splits_json) as f:
#             splits_payload = json.load(f)
#         text_splits = splits_payload.get('text_splits')
#         if text_splits is None:
#             print(f'[SETUP] {args.splits_json} has no "text_splits" — that checkpoint wasn\'t trained with '
#                   f'--use-word-trials/--use-sentence-trials, so no word trial was used for training; '
#                   f'evaluating every word trial found (all genuinely held-out).')
#         else:
#             test_only_dirs = {Path(trial_dir).resolve() for trial_dir, _ in text_splits['test']}
#             print(f'[SETUP] restricting evaluation to the {len(test_only_dirs)} held-out test trials in '
#                   f'{args.splits_json} (excludes trials this checkpoint was trained/validated on)')
#         letter_splits = splits_payload.get('splits')
#         if args.include_letters and letter_splits is not None and 'test' in letter_splits:
#             letter_test_only_dirs = {Path(trial_dir).resolve() for trial_dir, _ in letter_splits['test']}
#             print(f'[SETUP] restricting letter evaluation to the {len(letter_test_only_dirs)} held-out '
#                   f'letter test trials in {args.splits_json}')

#     participant_dirs = discover_participant_dataset_dirs(args.participants_root, args.participants)
#     if not participant_dirs:
#         raise RuntimeError(f'no participant dataset folders found under {args.participants_root}')
#     print(f'[SETUP] participants: {[name for name, _ in participant_dirs]}')

#     rows = []
#     # Scan sentence trials whenever the user explicitly asked for them
#     # (--include-sentences), OR whenever a splits.json test split might
#     # contain some regardless (see that block's own comment below) —
#     # either way this is just one flag deciding whether the sentence_dir
#     # glob below ever runs.
#     scan_sentences = args.include_sentences or (test_only_dirs is not None)
#     with torch.no_grad():
#         for participant_name, pdir in participant_dirs:
#             word_dir = pdir / args.word_subfolder
#             trial_dirs = sorted(p for p in word_dir.glob('trial_*') if p.is_dir()) if word_dir.exists() else []
#             n_word = len(trial_dirs)
#             n_sentence = 0
#             if scan_sentences:
#                 sentence_dir = pdir / args.sentence_subfolder
#                 if sentence_dir.exists():
#                     sentence_trial_dirs = sorted(p for p in sentence_dir.glob('trial_*') if p.is_dir())
#                     n_sentence = len(sentence_trial_dirs)
#                     trial_dirs += sentence_trial_dirs
#             if test_only_dirs is not None:
#                 # A test split built from --use-word-trials + --use-sentence-trials pools both
#                 # together before splitting (see train_ctc.py), so this split's test set can
#                 # contain either kind regardless of --include-sentences — restricting to it here
#                 # applies to whichever trial_dirs were scanned above.
#                 trial_dirs = [t for t in trial_dirs if t.resolve() in test_only_dirs]
#             print(f'[SETUP]   {participant_name}: {n_word} word trials in {word_dir}'
#                   + (f', {n_sentence} sentence trials in {pdir / args.sentence_subfolder}' if scan_sentences else ''))

#             for trial_dir in trial_dirs:
#                 info_path = trial_dir / 'trial_info.json'
#                 if not info_path.exists():
#                     continue
#                 with open(info_path) as f:
#                     info = json.load(f)
#                 true_word = info.get('content', '').strip().lower()
#                 if not use_space:
#                     # A checkpoint trained with --no-space (see
#                     # dataset_ctc_realtext.py's use_space) never had spaces
#                     # in ITS training targets — the model was asked to
#                     # spell "hecalledseventimes", not "he called seven
#                     # times", and its classifier has no way to output a
#                     # space at all. Comparing its prediction against the
#                     # ORIGINAL spaced ground truth would count every
#                     # missing space as an error the model was never in a
#                     # position to avoid, which isn't what a fair test of
#                     # "does removing the space class help LETTER
#                     # recognition" should measure — so the ground truth is
#                     # stripped of spaces here too, matching what the model
#                     # was actually trained/asked to reproduce. Word trials
#                     # are unaffected (their content never has spaces).
#                     true_word = true_word.replace(' ', '')
#                 if not true_word:
#                     continue

#                 span = get_word_span(trial_dir)
#                 if span is None:
#                     continue
#                 t_start, t_end = span

#                 audio = None
#                 imu = None
#                 if modality in ('audio', 'fusion'):
#                     audio = load_audio_variable(trial_dir, audio_source, t_start, t_end).unsqueeze(0).to(device)
#                 if modality in ('imu', 'fusion'):
#                     imu = load_imu_variable(trial_dir, imu_source, finger, t_start, t_end).unsqueeze(0).to(device)
#                 audio_len = torch.tensor([audio.shape[3]]).to(device) if audio is not None else None
#                 imu_len = torch.tensor([imu.shape[2]]).to(device) if imu is not None else None

#                 log_probs, out_len, _imu_recon, _spec_recon = model(audio, audio_len, imu, imu_len)
#                 if args.decode == 'beam':
#                     raw_pred = ctc_beam_search_decode(log_probs, out_len, classes,
#                                                        beam_width=args.beam_width, use_space=use_space)[0]
#                 else:
#                     raw_pred = ctc_greedy_decode(log_probs, out_len, classes, use_space=use_space)[0]
#                 raw_ed = edit_distance(raw_pred, true_word)
#                 is_sentence = trial_dir.parent.name == args.sentence_subfolder

#                 row = {
#                     'participant': participant_name, 'writing_target': 'sentence' if is_sentence else 'word',
#                     'trial': str(trial_dir), 'true_word': true_word,
#                     'span_sec': round(t_end - t_start, 3),
#                     'raw_ctc_pred': raw_pred, 'raw_correct': raw_pred == true_word,
#                     'raw_edit_distance': raw_ed,
#                 }
#                 if word_dictionary is not None:
#                     # A trial's own parent folder name tells us which pool/LM
#                     # to correct against — 'word' trials snap to the word
#                     # list (letter-bigram context within one word),
#                     # 'sentence' trials snap to the FULL SENTENCE list
#                     # instead (see autocorrect_ctc.py's module docstring for
#                     # why matching whole sentences is how this captures
#                     # neighboring-word context without a separate word-
#                     # segmentation step).
#                     candidates = sentence_dictionary if is_sentence else word_dictionary
#                     lm = sentence_lm if is_sentence else word_lm
#                     corrected_pred = autocorrect(raw_pred, candidates, lm)
#                     row.update({
#                         'dict_snapped_pred': corrected_pred, 'dict_correct': corrected_pred == true_word,
#                         'dict_edit_distance': edit_distance(corrected_pred, true_word),
#                     })
#                 rows.append(row)

#                 if not args.quiet:
#                     mark = 'OK  ' if raw_pred == true_word else '    '
#                     pred_shown = raw_pred if raw_pred else '(blank)'
#                     tag = 'S' if is_sentence else 'W'
#                     line = (f'  [{mark}][{tag}] {participant_name}  "{true_word}"({len(true_word)}) -> '
#                             f'"{pred_shown}"({len(raw_pred)})  edit={raw_ed}  span={t_end - t_start:.2f}s')
#                     if word_dictionary is not None:
#                         line += f'  | corrected->"{corrected_pred}"' + (' OK' if corrected_pred == true_word else '')
#                     print(line)

#             if args.include_letters:
#                 # Letter trials: dataset/<participant>/dataset/<class>/trial_XXX/
#                 # — one folder per class, exactly matching train_ctc.py's
#                 # LetterDatasetCTC (which this mirrors: full trial,
#                 # untrimmed — see dataset_ctc.LetterDatasetCTC.__getitem__,
#                 # which never crops to touch_on/off the way word/sentence
#                 # trials do below via get_word_span). Ground truth is the
#                 # class name itself (the parent folder), not a
#                 # trial_info.json — letter trials never have one.
#                 letter_dirs = []
#                 for cls in classes:
#                     cls_dir = pdir / cls
#                     if cls_dir.exists():
#                         letter_dirs += [(p, cls) for p in sorted(cls_dir.glob('trial_*')) if p.is_dir()]
#                 if letter_test_only_dirs is not None:
#                     letter_dirs = [(p, cls) for p, cls in letter_dirs if p.resolve() in letter_test_only_dirs]
#                 print(f'[SETUP]   {participant_name}: {len(letter_dirs)} letter trials')

#                 for trial_dir, true_letter in letter_dirs:
#                     audio = None
#                     imu = None
#                     if modality in ('audio', 'fusion'):
#                         audio = load_audio_variable(trial_dir, audio_source).unsqueeze(0).to(device)
#                     if modality in ('imu', 'fusion'):
#                         imu = load_imu_variable(trial_dir, imu_source, finger).unsqueeze(0).to(device)
#                     audio_len = torch.tensor([audio.shape[3]]).to(device) if audio is not None else None
#                     imu_len = torch.tensor([imu.shape[2]]).to(device) if imu is not None else None

#                     log_probs, out_len, _imu_recon, _spec_recon = model(audio, audio_len, imu, imu_len)
#                     if args.decode == 'beam':
#                         raw_pred = ctc_beam_search_decode(log_probs, out_len, classes,
#                                                            beam_width=args.beam_width, use_space=use_space)[0]
#                     else:
#                         raw_pred = ctc_greedy_decode(log_probs, out_len, classes, use_space=use_space)[0]
#                     raw_ed = edit_distance(raw_pred, true_letter)
#                     rows.append({
#                         'participant': participant_name, 'writing_target': 'letter',
#                         'trial': str(trial_dir), 'true_word': true_letter, 'span_sec': None,
#                         'raw_ctc_pred': raw_pred, 'raw_correct': raw_pred == true_letter,
#                         'raw_edit_distance': raw_ed,
#                     })
#                     if not args.quiet:
#                         mark = 'OK  ' if raw_pred == true_letter else '    '
#                         pred_shown = raw_pred if raw_pred else '(blank)'
#                         print(f'  [{mark}][L] {participant_name}  "{true_letter}" -> '
#                               f'"{pred_shown}"  edit={raw_ed}')

#     return pd.DataFrame(rows)


# def main():
#     args = build_arg_parser().parse_args()
#     df = run_evaluation(args)
#     args.out_dir.mkdir(parents=True, exist_ok=True)
#     df.to_csv(args.out_dir / 'word_ctc_results.csv', index=False)

#     n_letter = (df['writing_target'] == 'letter').sum()
#     n_word = (df['writing_target'] == 'word').sum()
#     n_sentence = (df['writing_target'] == 'sentence').sum()
#     print(f'\n[RESULT] {len(df)} trials evaluated ({n_letter} letter, {n_word} word, {n_sentence} sentence)')
#     # Letter/word/sentence trials are reported as SEPARATE blocks, never
#     # averaged together — they differ structurally (a letter target is a
#     # single character; a sentence is dozens, with spaces WER even
#     # applies to), so a single combined number would mostly just reflect
#     # the letter:word:sentence mix of whatever happened to be scanned,
#     # not model quality either way.
#     has_dictionary = 'dict_correct' in df.columns
#     report = {
#         'checkpoint': str(args.checkpoint),
#         'decode': args.decode,
#         'has_dictionary_correction': has_dictionary,
#     }
#     for target in ('letter', 'word', 'sentence'):
#         sub = df[df['writing_target'] == target]
#         if len(sub) == 0:
#             continue
#         print(f'\n[RESULT] === {target.upper()} trials ({len(sub)}) ===')
#         _print_summary_block(sub, has_dictionary=has_dictionary)
#         report[target] = _summarize_metrics(df, target, has_dictionary)

#     report_path = args.out_dir / 'report.json'
#     with open(report_path, 'w') as f:
#         json.dump(report, f, indent=2)
#     print(f'\n[DATA] {args.out_dir}/word_ctc_results.csv  (per-trial raw predictions)')
#     print(f'[DATA] {report_path}  (letter/word/sentence CER, accuracy, WER — for paper reporting)')


# def _print_summary_block(df, has_dictionary: bool):
#     """Prints the same [RESULT] metrics block for whichever subset of
#     rows it's given — factored out so word and sentence trials get
#     IDENTICAL reporting, just on their own separate slice of the data
#     (see main()'s per-writing_target loop)."""
#     n_blank = (df['raw_ctc_pred'] == '').sum()
#     true_lens = df['true_word'].str.len()
#     pred_lens = df['raw_ctc_pred'].str.len()
#     print(f'[RESULT] raw CTC exact-match accuracy:       {df["raw_correct"].mean():.3f}')
#     print(f'[RESULT] raw CTC mean edit distance:         {df["raw_edit_distance"].mean():.2f} characters')
#     # CER/WER (see their own docstrings for why exact-match accuracy
#     # alone is a poor way to judge decode quality — it scores "missed by
#     # one letter" identically to "completely wrong").
#     print(f'[RESULT] raw CER (Character Error Rate):     {_cer(df, "raw_ctc_pred", "raw_edit_distance"):.3f}')
#     raw_wer = _wer(df, 'raw_ctc_pred')
#     if not pd.isna(raw_wer):
#         print(f'[RESULT] raw WER (Word Error Rate):          {raw_wer:.3f}')
#     print(f'[RESULT] blank-only predictions:             {n_blank}/{len(df)} ({n_blank / len(df) * 100:.0f}%) '
#           f'— i.e. the model output nothing at all for these')
#     print(f'[RESULT] true text length:  mean={true_lens.mean():.1f}  min={true_lens.min()}  max={true_lens.max()}')
#     print(f'[RESULT] predicted length:  mean={pred_lens.mean():.1f}  min={pred_lens.min()}  max={pred_lens.max()}')
#     if pred_lens.mean() < true_lens.mean() * 0.5:
#         print('[NOTE] predictions are much SHORTER than the true text on average — the model is likely '
#               'still under-outputting (close to the "blank collapse" phase seen during training on much '
#               'shorter single-letter trials — see train_ctc.py\'s per-epoch diagnostics) rather than '
#               'confidently spelling out wrong letters.')
#     if has_dictionary:
#         print(f'[RESULT] context-aware corrected accuracy:   {df["dict_correct"].mean():.3f}')
#         print(f'[RESULT] context-aware corrected mean edit.: {df["dict_edit_distance"].mean():.2f} characters')
#         print(f'[RESULT] corrected CER:                      '
#               f'{_cer(df, "dict_snapped_pred", "dict_edit_distance"):.3f}')
#         corrected_wer = _wer(df, 'dict_snapped_pred')
#         if not pd.isna(corrected_wer):
#             print(f'[RESULT] corrected WER:                      {corrected_wer:.3f}')
#         print('[RESULT] per-participant context-aware corrected accuracy:')
#         print(df.groupby('participant')['dict_correct'].mean().to_string())
#     else:
#         print('[RESULT] per-participant raw CTC accuracy:')
#         print(df.groupby('participant')['raw_correct'].mean().to_string())


# if __name__ == '__main__':
#     main()

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


def _word_edit_distance(a_words: list, b_words: list) -> int:
    """Same Levenshtein recurrence as edit_distance() above, just
    operating on a list of WORDS instead of a string of CHARACTERS —
    used for WER below."""
    dp = list(range(len(b_words) + 1))
    for i, wa in enumerate(a_words, 1):
        prev, dp[0] = dp[0], i
        for j, wb in enumerate(b_words, 1):
            prev, dp[j] = dp[j], min(dp[j] + 1, dp[j - 1] + 1, prev + (wa != wb))
    return dp[-1]


def _cer(df, pred_col: str, ed_col: str) -> float:
    """Character Error Rate, MICRO-averaged: total edit distance summed
    across every trial, divided by the total number of ground-truth
    characters summed across every trial — NOT the mean of each trial's
    own (edit_distance / length) ratio, and NOT the same thing as mean
    edit distance (which weighs a 3-letter word's edit=1 the same as an
    11-letter word's edit=1, even though the relative error is very
    different). This is the standard metric OCR/ASR/handwriting-
    recognition papers report for exactly the reason exact-match
    accuracy alone doesn't capture: two predictions that are both
    "wrong" by exact-match can be very different in how close they
    actually got, and CER is what quantifies that — e.g. a 21-character
    sentence off by only 5 characters scores 0 on exact-match but
    ~0.24 CER, which is the more informative number for judging how
    good the model's decode actually was."""
    total_chars = df['true_word'].str.len().sum()
    if total_chars == 0:
        return float('nan')
    return df[ed_col].sum() / total_chars


def _wer(df, pred_col: str) -> float:
    """Word Error Rate, same micro-averaging idea as CER but at word
    granularity (split on whitespace) instead of character granularity —
    complements CER: a model can have a low CER (mostly right letters)
    while still getting very few WHOLE words exactly right if its
    mistakes are spread across many different words, which CER alone
    doesn't distinguish. Only meaningful when the ground truth actually
    contains spaces — returns NaN otherwise (word trials never have
    spaces in their content, and a --no-space checkpoint's sentence rows
    have their spaces stripped before this DataFrame is ever built — see
    run_evaluation's use_space handling)."""
    if not df['true_word'].str.contains(' ').any():
        return float('nan')
    total_words = 0
    total_word_edits = 0
    for true_text, pred_text in zip(df['true_word'], df[pred_col]):
        true_words = true_text.split()
        pred_words = pred_text.split()
        total_words += len(true_words)
        total_word_edits += _word_edit_distance(true_words, pred_words)
    return total_word_edits / total_words if total_words else float('nan')


def _summarize_metrics(df: pd.DataFrame, target: str, has_dictionary: bool) -> dict:
    """Flattens one writing_target's ('letter'/'word'/'sentence') slice
    of the results into a single dict of the metrics a paper would
    actually report: exact-match accuracy, CER, and (where meaningful —
    see _wer's own docstring on why this is NaN for letters/most words)
    WER, raw and dictionary-corrected. This is what report.json's
    per-target blocks are built from — see main()."""
    sub = df[df['writing_target'] == target]
    if len(sub) == 0:
        return {}
    out = {
        'n': int(len(sub)),
        'raw_accuracy': round(float(sub['raw_correct'].mean()), 4),
        'raw_cer': round(float(_cer(sub, 'raw_ctc_pred', 'raw_edit_distance')), 4),
        'raw_wer': round(float(_wer(sub, 'raw_ctc_pred')), 4) if not pd.isna(_wer(sub, 'raw_ctc_pred')) else None,
        'raw_mean_edit_distance': round(float(sub['raw_edit_distance'].mean()), 4),
    }
    if has_dictionary and 'dict_correct' in sub.columns:
        wer_corrected = _wer(sub, 'dict_snapped_pred')
        out.update({
            'corrected_accuracy': round(float(sub['dict_correct'].mean()), 4),
            'corrected_cer': round(float(_cer(sub, 'dict_snapped_pred', 'dict_edit_distance')), 4),
            'corrected_wer': round(float(wer_corrected), 4) if not pd.isna(wer_corrected) else None,
            'corrected_mean_edit_distance': round(float(sub['dict_edit_distance'].mean()), 4),
        })
    return out


def build_arg_parser():
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
    parser.add_argument('--include-letters', action='store_true',
                         help='also evaluate individual LETTER trials (dataset/<participant>/dataset/<class>/'
                              'trial_XXX/ - the same single-letter data train_ctc.py\'s LetterDatasetCTC '
                              'trains on), reported as its own separate [RESULT] block. When --splits-json '
                              'is given, restricted to that checkpoint\'s held-out letter test split the '
                              'same way word/sentence trials are restricted to the text_splits test set.')
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
    return parser


def run_evaluation(args) -> pd.DataFrame:
    """Does the actual work — load checkpoint, scan trials, decode,
    score — and returns the raw per-trial results as a DataFrame (same
    rows that get saved to word_ctc_results.csv). Factored out of main()
    specifically so compare_checkpoints.py can call this directly for
    several checkpoints in a row and build a side-by-side comparison
    table, without needing to re-implement or shell out to this
    evaluation logic. main() below is now a thin wrapper: call this, then
    print/save exactly what it already did."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    classes = ckpt['classes']
    modality = ckpt.get('modality', 'imu')
    rnn_hidden = ckpt.get('rnn_hidden', 256)   # older checkpoints predating the TCN/U-Net rewrite
                                                 # don't have this key — 256 matches train_ctc.py's own
                                                 # current default, so this only matters for checkpoints
                                                 # trained with a non-default --rnn-hidden
    use_space = ckpt.get('use_space', True)    # older checkpoints predating the space class always
                                                 # had one implicitly — True matches that
    sequence_encoder = ckpt.get('sequence_encoder', 'gru')   # older checkpoints predating this option
                                                                # were always GRU
    transformer_layers = ckpt.get('transformer_layers', 2)
    transformer_nhead = ckpt.get('transformer_nhead', 8)
    motion_target = ckpt.get('motion_target', 'fingertip_imu')   # older checkpoints predating
                                                                     # --motion-target were always this
                                                                     # (the IMU decoder's output width —
                                                                     # 6 vs 2 channels — depends on it, so
                                                                     # this MUST match how the checkpoint
                                                                     # was actually trained or
                                                                     # load_state_dict below fails on a
                                                                     # shape mismatch)
    audio_source = ckpt.get('audio_source', 'surface')
    imu_source = ckpt.get('imu_source', 'fingertip')
    finger = ckpt.get('finger', 'index')
    print(f'[SETUP] checkpoint: modality={modality} audio_source={audio_source} '
          f'imu_source={imu_source} classes={classes} use_space={use_space} '
          f'sequence_encoder={sequence_encoder} motion_target={motion_target}')
    decode_desc = f'beam search (width={args.beam_width})' if args.decode == 'beam' else 'greedy'
    print(f'[SETUP] decoding: {decode_desc}')

    model = build_model_ctc(modality, n_classes=len(classes), rnn_hidden=rnn_hidden,
                             use_space=use_space, sequence_encoder=sequence_encoder,
                             transformer_layers=transformer_layers,
                             transformer_nhead=transformer_nhead,
                             motion_target=motion_target).to(device)
    # strict=False: audio_only_classifier (see model_ctc.py's own docstring —
    # a training-only auxiliary head for real knowledge distillation, added
    # AFTER many of this project's own checkpoints were trained, never used
    # at inference/decode time) may legitimately be absent from an OLDER
    # checkpoint's own saved state_dict. That specific absence is safe to
    # ignore — but ANYTHING else missing/unexpected would mean a genuine
    # architecture mismatch between this checkpoint and the current
    # model_ctc.py, which must NOT be silently ignored (could easily produce
    # a model that loads without error but decodes garbage) — see the chat
    # this safeguard was added in.
    load_result = model.load_state_dict(ckpt['model_state_dict'], strict=False)
    _SAFE_TO_MISS = {'audio_only_classifier.weight', 'audio_only_classifier.bias'}
    unexpected_missing = set(load_result.missing_keys) - _SAFE_TO_MISS
    if unexpected_missing or load_result.unexpected_keys:
        raise RuntimeError(
            f'checkpoint/model_ctc.py architecture mismatch beyond the known-safe '
            f'audio_only_classifier case — missing: {unexpected_missing or "(none)"}, '
            f'unexpected: {load_result.unexpected_keys or "(none)"}. This means the current '
            f'model_ctc.py in this folder does not match how this checkpoint was actually '
            f'trained — do not proceed; use the exact model_ctc.py version this checkpoint '
            'was trained with instead.')
    if load_result.missing_keys:
        print('[SETUP] checkpoint predates audio_only_classifier (a training-only '
              'distillation head — see model_ctc.py) — left at its random init, which is fine '
              'since it is never used for decoding/evaluation.')
    model.eval()

    word_dictionary = None
    sentence_dictionary = None
    word_lm = None
    sentence_lm = None
    if args.phrase_set_path is not None:
        import phrase_set   # only needs to exist/import cleanly when actually used — see this
                             # module's docstring for why --phrase-set-path is optional at all
        phrase_set._load(args.phrase_set_path)
        word_dictionary = list(phrase_set._words)
        sentence_dictionary = list(phrase_set._phrases)
        if not use_space:
            # A --no-space checkpoint's raw predictions NEVER contain
            # spaces (see this file's own true_word-stripping above, a
            # few lines down in the per-trial loop) — comparing them via
            # edit distance against space-containing sentence candidates
            # systematically inflates every candidate's distance by
            # however many spaces it has, which can push even the TRUE
            # matching sentence far down (or off) the top-k shortlist
            # autocorrect() ranks from (see autocorrect_ctc.py). Left
            # unstripped, this made corrected sentence accuracy a flat
            # 0% across every single participant — not "worse than raw",
            # completely broken — see the chat this was fixed in.
            # Stripped here so candidates are compared on the exact same
            # character-for-character basis the model actually produces,
            # matching true_word's own stripping.
            sentence_dictionary = [s.replace(' ', '') for s in sentence_dictionary]
        # Separate bigram LMs for words vs. sentences — a sentence's own
        # character statistics (includes spaces, longer strings) are
        # different enough from a single word's that mixing them would
        # blur both. See autocorrect_ctc.py's module docstring for why
        # matching a raw sentence decode against the pool of REAL
        # sentences (rather than word-by-word) is how this captures
        # "neighboring word" context without a separate word-segmentation
        # step.
        word_lm = CharBigramLM(word_dictionary)
        sentence_lm = CharBigramLM(sentence_dictionary)
        print(f'[SETUP] dictionary: {len(word_dictionary)} words, {len(sentence_dictionary)} sentences '
              f'(context-aware autocorrected accuracy will be computed)')
    else:
        print('[SETUP] no --phrase-set-path given — reporting raw CTC accuracy only')

    test_only_dirs = None   # None = evaluate every word trial found; a set = restrict to just these
    letter_test_only_dirs = None   # same idea, for --include-letters — a SEPARATE set, since letter
                                     # trials are split via save_splits()'s own "splits" key, not
                                     # "text_splits" (see dataset.py's save_splits/make_splits)
    if args.splits_json is not None:
        with open(args.splits_json) as f:
            splits_payload = json.load(f)
        text_splits = splits_payload.get('text_splits')
        if text_splits is None:
            print(f'[SETUP] {args.splits_json} has no "text_splits" — that checkpoint wasn\'t trained with '
                  f'--use-word-trials/--use-sentence-trials, so no word trial was used for training; '
                  f'evaluating every word trial found (all genuinely held-out).')
        else:
            test_only_dirs = {Path(trial_dir).resolve() for trial_dir, _ in text_splits['test']}
            print(f'[SETUP] restricting evaluation to the {len(test_only_dirs)} held-out test trials in '
                  f'{args.splits_json} (excludes trials this checkpoint was trained/validated on)')
        letter_splits = splits_payload.get('splits')
        if args.include_letters and letter_splits is not None and 'test' in letter_splits:
            letter_test_only_dirs = {Path(trial_dir).resolve() for trial_dir, _ in letter_splits['test']}
            print(f'[SETUP] restricting letter evaluation to the {len(letter_test_only_dirs)} held-out '
                  f'letter test trials in {args.splits_json}')

    participant_dirs = discover_participant_dataset_dirs(args.participants_root, args.participants)
    if not participant_dirs:
        raise RuntimeError(f'no participant dataset folders found under {args.participants_root}')
    print(f'[SETUP] participants: {[name for name, _ in participant_dirs]}')

    rows = []
    # Scan sentence trials whenever the user explicitly asked for them
    # (--include-sentences), OR whenever a splits.json test split might
    # contain some regardless (see that block's own comment below) —
    # either way this is just one flag deciding whether the sentence_dir
    # glob below ever runs.
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
                # A test split built from --use-word-trials + --use-sentence-trials pools both
                # together before splitting (see train_ctc.py), so this split's test set can
                # contain either kind regardless of --include-sentences — restricting to it here
                # applies to whichever trial_dirs were scanned above.
                trial_dirs = [t for t in trial_dirs if t.resolve() in test_only_dirs]
            print(f'[SETUP]   {participant_name}: {n_word} word trials in {word_dir}'
                  + (f', {n_sentence} sentence trials in {pdir / args.sentence_subfolder}' if scan_sentences else ''))

            for trial_dir in trial_dirs:
                info_path = trial_dir / 'trial_info.json'
                if not info_path.exists():
                    continue
                with open(info_path) as f:
                    info = json.load(f)
                true_word = info.get('content', '').strip().lower()
                if not use_space:
                    # A checkpoint trained with --no-space (see
                    # dataset_ctc_realtext.py's use_space) never had spaces
                    # in ITS training targets — the model was asked to
                    # spell "hecalledseventimes", not "he called seven
                    # times", and its classifier has no way to output a
                    # space at all. Comparing its prediction against the
                    # ORIGINAL spaced ground truth would count every
                    # missing space as an error the model was never in a
                    # position to avoid, which isn't what a fair test of
                    # "does removing the space class help LETTER
                    # recognition" should measure — so the ground truth is
                    # stripped of spaces here too, matching what the model
                    # was actually trained/asked to reproduce. Word trials
                    # are unaffected (their content never has spaces).
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

                log_probs, out_len, _imu_recon, _spec_recon, _audio_only_log_probs = model(audio, audio_len, imu, imu_len)
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
                    # A trial's own parent folder name tells us which pool/LM
                    # to correct against — 'word' trials snap to the word
                    # list (letter-bigram context within one word),
                    # 'sentence' trials snap to the FULL SENTENCE list
                    # instead (see autocorrect_ctc.py's module docstring for
                    # why matching whole sentences is how this captures
                    # neighboring-word context without a separate word-
                    # segmentation step).
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

            if args.include_letters:
                # Letter trials: dataset/<participant>/dataset/<class>/trial_XXX/
                # — one folder per class, exactly matching train_ctc.py's
                # LetterDatasetCTC (which this mirrors: full trial,
                # untrimmed — see dataset_ctc.LetterDatasetCTC.__getitem__,
                # which never crops to touch_on/off the way word/sentence
                # trials do below via get_word_span). Ground truth is the
                # class name itself (the parent folder), not a
                # trial_info.json — letter trials never have one.
                letter_dirs = []
                for cls in classes:
                    cls_dir = pdir / cls
                    if cls_dir.exists():
                        letter_dirs += [(p, cls) for p in sorted(cls_dir.glob('trial_*')) if p.is_dir()]
                if letter_test_only_dirs is not None:
                    letter_dirs = [(p, cls) for p, cls in letter_dirs if p.resolve() in letter_test_only_dirs]
                print(f'[SETUP]   {participant_name}: {len(letter_dirs)} letter trials')

                for trial_dir, true_letter in letter_dirs:
                    audio = None
                    imu = None
                    if modality in ('audio', 'fusion'):
                        audio = load_audio_variable(trial_dir, audio_source).unsqueeze(0).to(device)
                    if modality in ('imu', 'fusion'):
                        imu = load_imu_variable(trial_dir, imu_source, finger).unsqueeze(0).to(device)
                    audio_len = torch.tensor([audio.shape[3]]).to(device) if audio is not None else None
                    imu_len = torch.tensor([imu.shape[2]]).to(device) if imu is not None else None

                    log_probs, out_len, _imu_recon, _spec_recon, _audio_only_log_probs = model(audio, audio_len, imu, imu_len)
                    if args.decode == 'beam':
                        raw_pred = ctc_beam_search_decode(log_probs, out_len, classes,
                                                           beam_width=args.beam_width, use_space=use_space)[0]
                    else:
                        raw_pred = ctc_greedy_decode(log_probs, out_len, classes, use_space=use_space)[0]
                    raw_ed = edit_distance(raw_pred, true_letter)
                    rows.append({
                        'participant': participant_name, 'writing_target': 'letter',
                        'trial': str(trial_dir), 'true_word': true_letter, 'span_sec': None,
                        'raw_ctc_pred': raw_pred, 'raw_correct': raw_pred == true_letter,
                        'raw_edit_distance': raw_ed,
                    })
                    if not args.quiet:
                        mark = 'OK  ' if raw_pred == true_letter else '    '
                        pred_shown = raw_pred if raw_pred else '(blank)'
                        print(f'  [{mark}][L] {participant_name}  "{true_letter}" -> '
                              f'"{pred_shown}"  edit={raw_ed}')

    return pd.DataFrame(rows)


def main():
    args = build_arg_parser().parse_args()
    df = run_evaluation(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dir / 'word_ctc_results.csv', index=False)

    n_letter = (df['writing_target'] == 'letter').sum()
    n_word = (df['writing_target'] == 'word').sum()
    n_sentence = (df['writing_target'] == 'sentence').sum()
    print(f'\n[RESULT] {len(df)} trials evaluated ({n_letter} letter, {n_word} word, {n_sentence} sentence)')
    # Letter/word/sentence trials are reported as SEPARATE blocks, never
    # averaged together — they differ structurally (a letter target is a
    # single character; a sentence is dozens, with spaces WER even
    # applies to), so a single combined number would mostly just reflect
    # the letter:word:sentence mix of whatever happened to be scanned,
    # not model quality either way.
    has_dictionary = 'dict_correct' in df.columns
    report = {
        'checkpoint': str(args.checkpoint),
        'decode': args.decode,
        'has_dictionary_correction': has_dictionary,
    }
    for target in ('letter', 'word', 'sentence'):
        sub = df[df['writing_target'] == target]
        if len(sub) == 0:
            continue
        print(f'\n[RESULT] === {target.upper()} trials ({len(sub)}) ===')
        _print_summary_block(sub, has_dictionary=has_dictionary)
        report[target] = _summarize_metrics(df, target, has_dictionary)

    report_path = args.out_dir / 'report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'\n[DATA] {args.out_dir}/word_ctc_results.csv  (per-trial raw predictions)')
    print(f'[DATA] {report_path}  (letter/word/sentence CER, accuracy, WER — for paper reporting)')


def _print_summary_block(df, has_dictionary: bool):
    """Prints the same [RESULT] metrics block for whichever subset of
    rows it's given — factored out so word and sentence trials get
    IDENTICAL reporting, just on their own separate slice of the data
    (see main()'s per-writing_target loop)."""
    n_blank = (df['raw_ctc_pred'] == '').sum()
    true_lens = df['true_word'].str.len()
    pred_lens = df['raw_ctc_pred'].str.len()
    print(f'[RESULT] raw CTC exact-match accuracy:       {df["raw_correct"].mean():.3f}')
    print(f'[RESULT] raw CTC mean edit distance:         {df["raw_edit_distance"].mean():.2f} characters')
    # CER/WER (see their own docstrings for why exact-match accuracy
    # alone is a poor way to judge decode quality — it scores "missed by
    # one letter" identically to "completely wrong").
    print(f'[RESULT] raw CER (Character Error Rate):     {_cer(df, "raw_ctc_pred", "raw_edit_distance"):.3f}')
    raw_wer = _wer(df, 'raw_ctc_pred')
    if not pd.isna(raw_wer):
        print(f'[RESULT] raw WER (Word Error Rate):          {raw_wer:.3f}')
    print(f'[RESULT] blank-only predictions:             {n_blank}/{len(df)} ({n_blank / len(df) * 100:.0f}%) '
          f'— i.e. the model output nothing at all for these')
    print(f'[RESULT] true text length:  mean={true_lens.mean():.1f}  min={true_lens.min()}  max={true_lens.max()}')
    print(f'[RESULT] predicted length:  mean={pred_lens.mean():.1f}  min={pred_lens.min()}  max={pred_lens.max()}')
    if pred_lens.mean() < true_lens.mean() * 0.5:
        print('[NOTE] predictions are much SHORTER than the true text on average — the model is likely '
              'still under-outputting (close to the "blank collapse" phase seen during training on much '
              'shorter single-letter trials — see train_ctc.py\'s per-epoch diagnostics) rather than '
              'confidently spelling out wrong letters.')
    if has_dictionary:
        print(f'[RESULT] context-aware corrected accuracy:   {df["dict_correct"].mean():.3f}')
        print(f'[RESULT] context-aware corrected mean edit.: {df["dict_edit_distance"].mean():.2f} characters')
        print(f'[RESULT] corrected CER:                      '
              f'{_cer(df, "dict_snapped_pred", "dict_edit_distance"):.3f}')
        corrected_wer = _wer(df, 'dict_snapped_pred')
        if not pd.isna(corrected_wer):
            print(f'[RESULT] corrected WER:                      {corrected_wer:.3f}')
        print('[RESULT] per-participant context-aware corrected accuracy:')
        print(df.groupby('participant')['dict_correct'].mean().to_string())
    else:
        print('[RESULT] per-participant raw CTC accuracy:')
        print(df.groupby('participant')['raw_correct'].mean().to_string())


if __name__ == '__main__':
    main()