"""
autocorrect_ctc.py
────────────────────────────────────────────────────────────────────────────
Replaces evaluate_word_ctc.py's plain nearest-edit-distance snap (pick
whichever dictionary word is closest, full stop) with a context-aware
version: candidates are scored by BOTH how close they are to the raw CTC
output (edit distance) AND how plausible they are as real written text (a
character-level bigram language model learned from the same MacKenzie
phrase set the dictionary itself comes from). Among several dictionary
words at the same edit distance from a garbled CTC decode, the one whose
letter sequence "reads like real English" is preferred over one that's
merely character-close but linguistically unlikely — this is the
"unsignificant edits" fix: minor misspellings get corrected using both
neighboring-LETTER context (the bigram model, within a word) and, for
sentence trials, effectively neighboring-WORD context too (see below).

For sentence trials specifically: since a real sentence already encodes
natural word-to-word transitions, the SAME scoring machinery snaps a raw
CTC sentence decode against the pool of real sentences (not individual
words) rather than word list — matching against actual sentences is a
simple, practical way to use "neighboring word" context without needing a
full word-segmentation-plus-word-level-language-model pipeline, which
would be a substantially larger undertaking. evaluate_word_ctc.py just
needs to pass the right candidate pool (word list vs. sentence list) for
whichever kind of trial is being decoded.
"""

import math
from collections import defaultdict


class CharBigramLM:
    """A smoothed character bigram model — P(next_char | prev_char) —
    trained by counting transitions across every string in `texts`
    (typically the dictionary's own words, or the phrase set's own
    sentences). Used only to judge how "plausible" a candidate is as real
    written text, independent of how close it happens to be to any
    particular raw CTC decode — see autocorrect() below for how the two
    are combined."""

    def __init__(self, texts: list, alphabet: str = 'abcdefghijklmnopqrstuvwxyz '):
        self.alphabet = alphabet
        self.counts = defaultdict(lambda: defaultdict(int))
        self.totals = defaultdict(int)
        start = '^'
        for text in texts:
            prev = start
            for c in text:
                if c not in alphabet:
                    continue
                self.counts[prev][c] += 1
                self.totals[prev] += 1
                prev = c
        self.vocab_size = len(alphabet)

    def log_prob(self, text: str) -> float:
        """Sum of log P(c_i | c_{i-1}) over the string, with add-1
        (Laplace) smoothing so a bigram never seen in training gets a
        small nonzero probability instead of zeroing out the whole
        candidate outright."""
        prev = '^'
        total = 0.0
        for c in text:
            count = self.counts[prev].get(c, 0)
            denom = self.totals.get(prev, 0)
            prob = (count + 1) / (denom + self.vocab_size)   # add-1 smoothing
            total += math.log(prob)
            prev = c
        return total


def edit_distance(a: str, b: str) -> int:
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, dp[0] = dp[0], i
        for j, cb in enumerate(b, 1):
            prev, dp[j] = dp[j], min(dp[j] + 1, dp[j - 1] + 1, prev + (ca != cb))
    return dp[-1]


def autocorrect(raw: str, candidates: list, lm: CharBigramLM, edit_weight: float = 1.0,
                 lm_weight: float = 0.3, top_k_by_edit: int = 20) -> str:
    """Picks the best candidate from `candidates` (a word list OR a
    sentence list — same function either way, see module docstring) for
    the given raw CTC decode `raw`, combining edit distance (how close
    the candidate is to what the model actually output) with the
    character bigram LM's log-likelihood (how plausible the candidate is
    as real text):

        score(candidate) = -edit_weight * edit_distance(raw, candidate)
                            + lm_weight  * lm.log_prob(candidate)

    Only the top_k_by_edit closest-by-edit-distance candidates are
    LM-rescored — evaluating every candidate's LM score is the expensive
    part of this function, and edit distance alone already narrows the
    field to plausible contenders first. This keeps the combined approach
    nearly as fast as the plain nearest-neighbor snap it replaces, while
    still letting the LM break ties/near-ties among the closest few."""
    if not candidates:
        return raw
    if not raw:
        # nothing to anchor edit distance to (e.g. the model output
        # nothing at all) — fall back to the single most probable string
        # under the LM alone
        return max(candidates, key=lm.log_prob)
    shortlist = sorted(candidates, key=lambda w: edit_distance(raw, w))[:top_k_by_edit]
    best = max(shortlist, key=lambda w: -edit_weight * edit_distance(raw, w) + lm_weight * lm.log_prob(w))
    return best
