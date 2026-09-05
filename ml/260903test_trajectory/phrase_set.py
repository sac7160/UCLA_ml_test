"""
collector/core/phrase_set.py
────────────────────────────────────────────────────────────────────────────
Loads the MacKenzie & Soukoreff (2003) 500-phrase set and derives the three
writing-target pools the final protocol needs (letter / word / sentence):

  letter    the 26 lowercase letters — independent of the phrase set itself.
  word      every unique word across all 500 phrases (already lowercase,
            no punctuation — that's the phrase set's own convention, see
            below), so "word" and "sentence" both trace back to the same
            source material, matching the protocol summary's "MacKenzie
            Phrase Set (all lower case)" note under both.
  sentence  the 500 phrases themselves, unmodified.

The phrase set itself is NOT bundled here — it's the authors' own public
resource, not something to fork a copy of into this repo. Download it
yourself:
    http://www.yorku.ca/mack/PhraseSets.zip
Unzip it, and place phrases2.txt at the path in config.PHRASE_SET_PATH
(one phrase per line, all-lowercase, no punctuation — exactly the format
the zip ships in; no preprocessing needed).

Usage:
    from core import phrase_set
    text = phrase_set.random_item('word', config.PHRASE_SET_PATH)
"""

import random
from pathlib import Path
from typing import Optional

LETTERS = list('abcdefghijklmnopqrstuvwxyz')

# Module-level cache: the phrase file is parsed once per process, not once
# per random_word()/random_sentence() call — trial-to-trial stimulus
# selection happens far more often than the file itself ever changes
# within a session.
_phrases: Optional[list] = None
_words: Optional[list] = None
_loaded_from: Optional[Path] = None


def _load(path: Path):
    global _phrases, _words, _loaded_from
    if _phrases is not None and _loaded_from == path:
        return
    if not path.exists():
        raise FileNotFoundError(
            f'MacKenzie phrase set not found at {path}. Download it from '
            f'http://www.yorku.ca/mack/PhraseSets.zip, unzip, and place '
            f'phrases2.txt there (see this module\'s docstring).'
        )
    with open(path) as f:
        phrases = [line.strip().lower() for line in f if line.strip()]
    if not phrases:
        raise ValueError(f'{path} exists but contains no usable phrases (empty after stripping blank lines)')

    seen = set()
    words = []
    for phrase in phrases:
        for w in phrase.split():
            if w not in seen:
                seen.add(w)
                words.append(w)

    _phrases, _words, _loaded_from = phrases, words, path


def random_letter(rng: Optional[random.Random] = None) -> str:
    r = rng or random
    return r.choice(LETTERS)


def random_word(path: Path, rng: Optional[random.Random] = None) -> str:
    _load(path)
    r = rng or random
    return r.choice(_words)


def random_sentence(path: Path, rng: Optional[random.Random] = None) -> str:
    _load(path)
    r = rng or random
    return r.choice(_phrases)


def random_item(writing_target: str, path: Path, rng: Optional[random.Random] = None) -> str:
    """writing_target: 'letter' / 'word' / 'sentence' — the one place that
    dispatches by writing-target type, so callers (the instructor window's
    "next stimulus" logic) don't need their own if/elif on it."""
    if writing_target == 'letter':
        return random_letter(rng)
    elif writing_target == 'word':
        return random_word(path, rng)
    elif writing_target == 'sentence':
        return random_sentence(path, rng)
    raise ValueError(f'writing_target must be "letter", "word", or "sentence" — got "{writing_target}"')


def pool_size(writing_target: str, path: Path) -> int:
    """How many distinct items are available for a given writing_target —
    useful for sanity-checking (e.g. logging "picking from N words") and
    for anything that wants to sample without replacement later."""
    if writing_target == 'letter':
        return len(LETTERS)
    _load(path)
    if writing_target == 'word':
        return len(_words)
    elif writing_target == 'sentence':
        return len(_phrases)
    raise ValueError(f'writing_target must be "letter", "word", or "sentence" — got "{writing_target}"')