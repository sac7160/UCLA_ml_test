"""
dataset_ctc_realtext.py
────────────────────────────────────────────────────────────────────────────
Uses the ACTUALLY COLLECTED word/sentence trials as CTC training examples
— not just for held-out evaluation (evaluate_word_ctc.py), but as
additional training signal alongside letter trials and the synthetic
concatenated-letter examples (dataset_ctc_concat.py).

Why this matters: dataset_ctc_concat.py's synthetic examples are letter
trials pasted together with an artificial silence gap between each — real
continuous handwriting has none of that (no silence between letters,
natural co-articulation as the finger moves from one letter's end
position into the next letter's start). A real word/sentence trial is the
closest thing to the model's actual test-time input the training set can
contain. Mixing real text trials in on top of the synthetic ones gives
the model exposure to both.

Ground truth: trial_info.json's `content` field — the exact same field
evaluate_word_ctc.py reads. Span: the same "first audio_touch_on to last
audio_touch_off" window get_word_span() uses (duplicated here rather than
imported from evaluate_word_ctc.py, to keep that file purely an
evaluation script with no role in training).

Sentences contain spaces between words; those aren't part of the 26-
letter class set, so they're simply dropped from the target sequence when
building it — the model is asked to spell the sentence's letters in
order, not to also mark where word boundaries fall.

IMPORTANT — held-out fairness: word/sentence trials used here for
TRAINING must not also be the ones evaluate_word_ctc.py reports numbers
on, or that accuracy would be inflated by testing on trials the model
was directly trained on. split_text_trials() below produces the same
train/val/test partition train_ctc.py already uses for letters (saved
into splits.json) — evaluate_word_ctc.py should be pointed at (or
filtered to) the 'test' portion specifically once this is in use; see
train_ctc.py's printed reminder when --use-word-trials/--use-sentence-
trials is set.
"""

import json
import random
import wave
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

import config_ctc
from dataset_ctc import load_audio_variable, load_imu_variable


def _wav_is_valid(path: Path) -> bool:
    """A failed/interrupted recording (e.g. the watch disconnected
    mid-trial) can leave a 0-frame wav file behind — it exists, so a
    plain exists()-check lets it through, but there's nothing to
    resample and torchaudio crashes trying (see the chat this was added
    in — this exact scenario took down a full training run partway
    through, right as a real word/sentence trial with a corrupted
    watch_audio.wav came up). Identical to dataset.py's own
    _wav_is_valid(), duplicated here rather than imported since dataset.py
    is the original fixed-length pipeline this file deliberately doesn't
    touch or depend on."""
    try:
        with wave.open(str(path), 'rb') as wf:
            return wf.getnframes() > 0
    except Exception:
        return False


def _csv_has_data_rows(path: Path) -> bool:
    """Same idea as _wav_is_valid, for the IMU CSV — a header line with
    nothing after it is a completed-but-empty file, not a missing one."""
    try:
        with open(path) as f:
            next(f)              # header
            return next(f, None) is not None
    except Exception:
        return False


def get_word_span(trial_dir: Path):
    """Returns (first_touch_on, last_touch_off) in trial-relative
    seconds, or None if events.csv has no complete on/off pair — same
    logic as evaluate_word_ctc.py's function of the same name (duplicated
    rather than imported; see this module's docstring for why)."""
    events_path = trial_dir / 'events.csv'
    if not events_path.exists():
        return None
    df = pd.read_csv(events_path).sort_values('time_aligned')
    on_times = df[df['event'] == 'audio_touch_on']['time_aligned']
    off_times = df[df['event'] == 'audio_touch_off']['time_aligned']
    if on_times.empty or off_times.empty:
        return None
    return float(on_times.iloc[0]), float(off_times.iloc[-1])


def scan_text_trials(participant_dataset_dir: Path, subfolder: str,
                      audio_source: str = config_ctc.DEFAULT_AUDIO_SOURCE,
                      imu_source: str = config_ctc.DEFAULT_IMU_SOURCE) -> list:
    """Returns [(trial_dir, text), ...] for every usable trial under
    participant_dataset_dir/subfolder/trial_XXX/ — subfolder is typically
    'word' or 'sentence' (see train_ctc.py's --word-subfolder/
    --sentence-subfolder). A trial is skipped (not usable) if it has no
    trial_info.json, an empty `content`, no valid touch_on/off span, or —
    same check dataset.py's scan_dataset() already does for letters, and
    the ONE this function was originally missing (see this module's
    _wav_is_valid()'s docstring for the crash that caused) — a 0-frame
    audio file or an IMU CSV with no data rows, either of which usually
    means the recording was interrupted partway through."""
    text_dir = participant_dataset_dir / subfolder
    if not text_dir.exists():
        return []
    audio_filename = config_ctc.AUDIO_FILENAMES[audio_source]
    imu_filename = config_ctc.IMU_FILENAMES[imu_source]
    results = []
    for trial_dir in sorted(text_dir.glob('trial_*')):
        if not trial_dir.is_dir():
            continue
        info_path = trial_dir / 'trial_info.json'
        if not info_path.exists():
            continue
        with open(info_path) as f:
            info = json.load(f)
        text = info.get('content', '').strip().lower()
        if not text:
            continue
        audio_path = trial_dir / audio_filename
        imu_path = trial_dir / imu_filename
        if not audio_path.exists() or not imu_path.exists():
            continue
        if not _wav_is_valid(audio_path):
            print(f'[WARN] {trial_dir}: {audio_filename} has 0 audio frames — skipping '
                  f'(likely an interrupted/failed recording)')
            continue
        if not _csv_has_data_rows(imu_path):
            print(f'[WARN] {trial_dir}: {imu_filename} has no data rows — skipping '
                  f'(likely an interrupted/failed recording)')
            continue
        if get_word_span(trial_dir) is None:
            continue
        results.append((trial_dir, text))
    return results


def split_text_trials(samples: list, train_frac: float, val_frac: float, test_frac: float, seed: int) -> dict:
    """A plain random split — NOT stratified by content the way
    dataset.py's make_splits() stratifies by letter class. With word/
    sentence content drawn close to uniformly from a ~1000+-word pool,
    almost every trial's text is unique, so there's no meaningful "class"
    per trial to stratify by; make_splits() would treat each unique text
    as its own 1-member class and degenerate to putting nearly everything
    in train (see the chat this was written in)."""
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6
    rng = random.Random(seed)
    shuffled = samples[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_test = round(n * test_frac)
    n_val = round(n * val_frac)
    return {
        'test': shuffled[:n_test],
        'val': shuffled[n_test:n_test + n_val],
        'train': shuffled[n_test + n_val:],
    }


class RealTextDatasetCTC(Dataset):
    """samples: [(trial_dir, text), ...] as returned by scan_text_trials()
    — text is the trial's ground-truth string (a word, or a sentence with
    spaces stripped out at target-encoding time, see __getitem__).
    `classes` must be the SAME letter list the model's classifier head
    was sized against (e.g. a-z) — any character in `text` that isn't in
    `classes` (spaces, punctuation, digits) is simply skipped when
    building the target, not an error.

    Output shape matches LetterDatasetCTC/ConcatLetterDatasetCTC exactly
    (mel, imu, target) — dataset_ctc.py's collate_fn_ctc works on this
    unmodified; all three can share one torch.utils.data.ConcatDataset."""

    def __init__(self, samples: list, classes: list,
                 finger: str = config_ctc.DEFAULT_FINGER,
                 audio_source: str = config_ctc.DEFAULT_AUDIO_SOURCE,
                 imu_source: str = config_ctc.DEFAULT_IMU_SOURCE):
        if not samples:
            raise RuntimeError('RealTextDatasetCTC got an empty sample list')
        self.samples = samples
        self.char_to_idx = {c: i for i, c in enumerate(classes)}
        self.finger = finger
        self.audio_source = audio_source
        self.imu_source = imu_source

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        trial_dir, text = self.samples[idx]
        t_start, t_end = get_word_span(trial_dir)
        audio = load_audio_variable(trial_dir, self.audio_source, t_start, t_end)
        imu = load_imu_variable(trial_dir, self.imu_source, self.finger, t_start, t_end)
        target = torch.tensor(
            [self.char_to_idx[c] + 1 for c in text if c in self.char_to_idx], dtype=torch.long)
        if len(target) == 0:
            # every character in `text` fell outside `classes` (shouldn't
            # normally happen for real English words against a-z, but
            # guards against e.g. a trial whose content is pure
            # punctuation/digits) — fall back to a single blank-adjacent
            # dummy target rather than returning a length-0 target, which
            # nn.CTCLoss cannot score
            target = torch.tensor([1], dtype=torch.long)
        return audio, imu, target
