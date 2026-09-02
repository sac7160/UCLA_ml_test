"""
train_writeas.py
────────────────────────────────────────────────────────────────────────────
Trains model_writeas.WriteASModel (see that file's own docstring) as a
baseline to compare against this project's own CTC architecture — same
data, same task, different model.

Trains on ALL THREE of this project's own trial types together — letter,
word, AND sentence — not word-only: this project's own word-only trial
count turned out too small on its own for this model to learn anything
useful (see the chat this was changed in: raw word-only training gave
~96% CTC CER). Letter trials (dataset/<participant>/dataset/<class>/
trial_XXX/, scanned via dataset.scan_dataset — the SAME original,
unmodified function train_ctc.py itself uses) are converted into
single-character "text" samples and pooled together with word/sentence
trials (dataset_ctc_realtext.scan_text_trials) into ONE combined training
set — see _letters_as_text_samples below. Sentence content has its
spaces stripped before being pooled in (see WriteASWordDataset's own
docstring on why: WriteAS's own alphabet, per the paper's own Section
3.1, is 26 lowercase letters only, with no space class — matching this
project's own established --no-space convention rather than inventing a
27th class WriteAS's own architecture was never sized for).

Loss: L_word = (1-lambda)*L_attention + lambda*L_CTC (paper's own Eq. 11,
lambda=0.8 default — the paper's own tuned value, Section 4.3.3).
L_attention is standard per-step cross-entropy against the teacher-forced
target sequence (letters + <EOS>, see collate_writeas below); L_CTC is
nn.CTCLoss against the same letters (no <EOS>/<BOS> — CTC has its own
blank symbol instead).

Usage:
    python train_writeas.py \\
        --participants-root ../dataset --classes a b c d e f g h i j k l m n o p q r s t u v w x y z \\
        --imu-source watch --word-subfolder word --sentence-subfolder sentence \\
        --epochs 100 --augment --word-concat-per-epoch 500 \\
        --out-dir checkpoints/writeas_baseline
"""

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config_ctc
from dataset import scan_dataset   # the ORIGINAL, unmodified letter-trial scanner — same
                                     # function train_ctc.py itself uses, reused here as-is
                                     # (see this file's own top-level docstring on why letters
                                     # are included now)
from dataset_ctc import load_imu_variable, discover_participant_dataset_dirs
from dataset_ctc_realtext import scan_text_trials, get_word_span
from model_writeas import WriteASModel
from writeas_augment import apply_random_augmentation


def _letters_as_text_samples(pdir: Path, classes: list, imu_source: str) -> list:
    """Converts scan_dataset's own [(trial_dir, int_label), ...] letter
    format into [(trial_dir, "x"), ...] — a length-1 STRING — so letter
    trials can be pooled directly into the same list format
    scan_text_trials returns for word/sentence trials (see main()),
    without WriteASWordDataset needing to know or care whether a given
    sample originally came from a letter, word, or sentence trial."""
    letter_samples = scan_dataset(pdir, classes, imu_source=imu_source)
    return [(trial_dir, classes[label]) for trial_dir, label in letter_samples]


class WriteASWordDataset(Dataset):
    """(imu, letter_indices) for each letter/word/sentence trial —
    letter_indices excludes spaces (WriteAS's own alphabet is 26
    lowercase letters only, Section 3.1: "All the 26 English
    characters... All the words are written in lowercase" — no space
    class in the paper's own task). A sentence's content simply has its
    spaces dropped before being turned into letter_indices — see this
    file's own top-level docstring."""

    def __init__(self, samples: list, classes: list, imu_source: str,
                 finger: str = config_ctc.DEFAULT_FINGER, augment: bool = False,
                 sample_rate: float = None):
        if not samples:
            raise RuntimeError('WriteASWordDataset got an empty sample list')
        self.samples = samples
        self.char_to_idx = {c: i for i, c in enumerate(classes)}
        self.imu_source = imu_source
        self.finger = finger
        # See writeas_augment.py's own module docstring — applies exactly
        # ONE of {none, time_warp, segment_warp, rotation_injection} per
        # __getitem__ call when True. Should only ever be True for a
        # TRAINING dataset — never the validation split (see main()),
        # matching the paper's own practice of evaluating on real,
        # unaugmented data only.
        self.augment = augment
        self.sample_rate = sample_rate or config_ctc.IMU_RESAMPLE_HZ

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        trial_dir, text = self.samples[idx]
        # get_word_span can legitimately return None — e.g. a letter
        # trial whose events.csv has no complete touch_on/off pair
        # (letters weren't originally scanned/trimmed this way before
        # letters were pooled in here — see this file's own top-level
        # docstring) — falls back to the WHOLE trial (no crop) rather
        # than crashing.
        span = get_word_span(trial_dir)
        t_start, t_end = span if span is not None else (None, None)
        imu = load_imu_variable(trial_dir, self.imu_source, self.finger, t_start, t_end)
        if self.augment:
            imu = apply_random_augmentation(imu, self.sample_rate)
        letters = [c for c in text if c in self.char_to_idx]
        if not letters:
            letters = [text[0]] if text and text[0] in self.char_to_idx else ['a']   # degenerate
                                                                                        # fallback —
                                                                                        # matches this
                                                                                        # project's own
                                                                                        # "never return
                                                                                        # an empty
                                                                                        # target" rule
        target = torch.tensor([self.char_to_idx[c] for c in letters], dtype=torch.long)
        return imu, target


class WriteASConcatDataset(Dataset):
    """Section 4.1's "word concatenation" augmentation: two samples FROM
    THE SAME PARTICIPANT (paper's own explicit restriction — "applied on
    the training samples from the same person") are joined end-to-end,
    both in their IMU signal and their letter-sequence label, to
    synthesize a new, longer multi-word training example — see the
    paper's own [xa, xb] -> [Wa, Wb] notation. This is a SEPARATE Dataset
    (not a mode of WriteASWordDataset) since it needs to draw TWO random
    picks per __getitem__ call, mirroring this project's own
    ConcatLetterDatasetCTC design for the same underlying reason
    (dataset_ctc_concat.py).

    n_synthetic_per_epoch controls how many concatenated examples are
    generated PER EPOCH (drawn fresh each time, not a fixed pre-built
    set) — matching this project's own established on-the-fly synthetic-
    augmentation pattern rather than the paper's literal "pre-generate a
    5x-larger dataset" framing (see writeas_augment.py's own top-level
    docstring)."""

    def __init__(self, participant_samples: dict, classes: list, imu_source: str,
                 n_synthetic_per_epoch: int, finger: str = config_ctc.DEFAULT_FINGER,
                 gap_sec: float = 0.3, seed: int = None):
        # participant_samples: {participant_name: [(trial_dir, text), ...]} — grouped by
        # participant so a concat pick can be restricted to the SAME person's own samples.
        self.participant_samples = {p: s for p, s in participant_samples.items() if len(s) >= 2}
        if not self.participant_samples:
            raise RuntimeError('WriteASConcatDataset needs at least one participant with >= 2 samples')
        self.participant_names = list(self.participant_samples.keys())
        self.char_to_idx = {c: i for i, c in enumerate(classes)}
        self.imu_source = imu_source
        self.finger = finger
        self.n_synthetic_per_epoch = n_synthetic_per_epoch
        self.gap_sec = gap_sec
        self.rng = random.Random(seed)

    def __len__(self):
        return self.n_synthetic_per_epoch

    def __getitem__(self, idx):
        participant = self.rng.choice(self.participant_names)
        pool = self.participant_samples[participant]
        trial_a, trial_b = self.rng.sample(pool, 2)   # two DISTINCT samples from the same person

        imus, texts = [], []
        for trial_dir, text in (trial_a, trial_b):
            span = get_word_span(trial_dir)   # can be None — see WriteASWordDataset.__getitem__'s
                                                 # own comment on why (letters, mainly)
            t_start, t_end = span if span is not None else (None, None)
            imus.append(load_imu_variable(trial_dir, self.imu_source, self.finger, t_start, t_end))
            texts.append(text)

        sample_rate = config_ctc.IMU_RESAMPLE_HZ
        gap_steps = max(1, int(round(self.gap_sec * sample_rate)))
        pieces = [imus[0], imus[0][:, -1:].repeat(1, gap_steps), imus[1]]   # hold-last-value gap —
                                                                               # same reasoning as this
                                                                               # project's own
                                                                               # dataset_ctc_concat.py
                                                                               # (a literal-zero gap
                                                                               # would look like an
                                                                               # unrealistic snap)
        full_imu = torch.cat(pieces, dim=1)

        letters = [c for text in texts for c in text if c in self.char_to_idx]
        if not letters:
            letters = ['a']
        target = torch.tensor([self.char_to_idx[c] for c in letters], dtype=torch.long)
        return full_imu, target


def collate_writeas(batch, n_classes: int, bos_idx: int, eos_idx: int):
    """Pads IMU to the batch's own max length (same convention as
    dataset_ctc.collate_fn_ctc) and builds TWO target views from the same
    letter sequence: ctc_targets (flat + lengths, for nn.CTCLoss, no
    BOS/EOS) and attention_input/attention_target (BOS-prefixed /
    EOS-suffixed, for teacher-forced cross-entropy — see train_writeas.py's
    own module docstring)."""
    imus, targets = zip(*batch)
    imu_lengths = torch.tensor([i.shape[1] for i in imus], dtype=torch.long)
    imu_max_t = int(imu_lengths.max())
    imu_padded = torch.stack([F.pad(i, (0, imu_max_t - i.shape[1])) for i in imus])

    ctc_target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    ctc_targets_concat = torch.cat(targets)

    max_len = max(len(t) for t in targets)
    B = len(targets)
    attn_input = torch.full((B, max_len + 1), eos_idx, dtype=torch.long)     # pad value irrelevant
                                                                                # past each sample's
                                                                                # own real length —
                                                                                # loss masks it out
    attn_target = torch.full((B, max_len + 1), eos_idx, dtype=torch.long)
    attn_lengths = torch.zeros(B, dtype=torch.long)
    for i, t in enumerate(targets):
        attn_input[i, 0] = bos_idx
        attn_input[i, 1:1 + len(t)] = t
        attn_target[i, :len(t)] = t
        attn_target[i, len(t)] = eos_idx
        attn_lengths[i] = len(t) + 1   # +1 for <EOS>

    return imu_padded, imu_lengths, ctc_targets_concat, ctc_target_lengths, \
        attn_input, attn_target, attn_lengths


def compute_attention_loss(attn_logits: torch.Tensor, attn_target: torch.Tensor,
                            attn_lengths: torch.Tensor) -> torch.Tensor:
    """Per-step cross-entropy, masked past each sample's own real length
    (letters + <EOS> — see collate_writeas)."""
    B, steps, n_classes = attn_logits.shape
    mask = torch.arange(steps, device=attn_logits.device).unsqueeze(0) < attn_lengths.unsqueeze(1)
    loss = F.cross_entropy(attn_logits.reshape(-1, n_classes), attn_target.reshape(-1), reduction='none')
    loss = loss.reshape(B, steps) * mask
    return loss.sum() / mask.sum().clamp(min=1)


def ctc_greedy_decode_writeas(ctc_log_probs: torch.Tensor, lengths: torch.Tensor, classes: list) -> list:
    """Same greedy collapse-repeats-then-drop-blanks logic as
    model_ctc.ctc_greedy_decode, standalone here since this baseline's
    blank index (0) and class ordering match this project's own
    convention but the two model files are otherwise independent."""
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


def build_arg_parser():
    parser = argparse.ArgumentParser(description='Train the WriteAS baseline replication (see '
                                                   'model_writeas.py) on this project\'s own word/'
                                                   'sentence trials.')
    parser.add_argument('--participants-root', type=Path, required=True)
    parser.add_argument('--participants', nargs='+', default=None)
    parser.add_argument('--classes', nargs='+', required=True)
    parser.add_argument('--imu-source', default=config_ctc.DEFAULT_IMU_SOURCE)
    parser.add_argument('--finger', default=config_ctc.DEFAULT_FINGER)
    parser.add_argument('--word-subfolder', default='word')
    parser.add_argument('--sentence-subfolder', default=None,
                         help='optional — sentence trials are only included if you explicitly ask, '
                              'and their spaces are simply dropped from the target the same way '
                              'this project\'s own --no-space mode does (WriteAS\'s own alphabet is '
                              '26 lowercase letters only, Section 3.1 — no space class).')
    parser.add_argument('--use-letter-trials', action='store_true',
                         help='also pool in single-letter trials (dataset/<participant>/dataset/'
                              '<class>/trial_XXX/, scanned via the ORIGINAL dataset.scan_dataset — '
                              'the same function train_ctc.py itself uses) as length-1 "text" '
                              'samples, alongside word/sentence trials. Recommended whenever word-'
                              'only data is too small on its own — see this file\'s own top-level '
                              'docstring.')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--ctc-attention-lambda', type=float, default=0.8,
                         help='paper\'s own Eq. 11 weighting: L_word = (1-lambda)*L_attention + '
                              'lambda*L_CTC. Paper\'s own tuned value is 0.8.')
    parser.add_argument('--train-frac', type=float, default=0.7)
    parser.add_argument('--val-frac', type=float, default=0.15)
    parser.add_argument('--test-frac', type=float, default=0.15,
                         help='held out and NEVER touched during training (not used for gradient '
                              'updates, not used to pick the best checkpoint) — saved to splits.json '
                              'so evaluate_writeas.py can later evaluate on genuinely unseen trials. '
                              'This is what makes --val-frac different from this: val trials ARE used '
                              'during training, just for checkpoint selection rather than gradients.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--augment', action='store_true',
                         help='enables time_warp/segment_warp/rotation_injection (see '
                              'writeas_augment.py), applied on-the-fly to TRAINING samples only — '
                              'paper Section 4.1/5.1.4. Never applied to the validation split.')
    parser.add_argument('--word-concat-per-epoch', type=int, default=0,
                         help='paper Section 4.1\'s "word concatenation" augmentation — generates '
                              'this many synthetic 2-word concatenated training examples PER EPOCH '
                              '(drawn fresh each epoch, restricted to same-participant pairs — see '
                              'WriteASConcatDataset). 0 (default) disables it entirely.')
    parser.add_argument('--out-dir', type=Path, default=Path('checkpoints/writeas_baseline'))
    return parser


def main():
    args = build_arg_parser().parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[SETUP] device={device}')

    participant_dirs = discover_participant_dataset_dirs(args.participants_root, args.participants)
    if not participant_dirs:
        raise RuntimeError(f'no participant dataset folders found under {args.participants_root}')

    samples = []
    samples_by_participant = {}   # needed for --word-concat-per-epoch (WriteASConcatDataset),
                                    # since paper Section 4.1 restricts concatenation to
                                    # SAME-participant pairs
    for name, pdir in participant_dirs:
        samples_by_participant[name] = []
        if args.use_letter_trials:
            s0 = _letters_as_text_samples(pdir, args.classes, args.imu_source)
            samples.extend(s0)
            samples_by_participant[name].extend(s0)
            print(f'[SETUP]   {name}: {len(s0)} letter trials')
        s = scan_text_trials(pdir, args.word_subfolder, imu_source=args.imu_source)
        samples.extend(s)
        samples_by_participant[name].extend(s)
        print(f'[SETUP]   {name}: {len(s)} word trials')
        if args.sentence_subfolder:
            s2 = scan_text_trials(pdir, args.sentence_subfolder, imu_source=args.imu_source)
            samples.extend(s2)
            samples_by_participant[name].extend(s2)
            print(f'[SETUP]   {name}: {len(s2)} sentence trials')
    if not samples:
        raise RuntimeError('no usable letter/word/sentence trials found')

    random.Random(args.seed).shuffle(samples)
    n_test = max(1, int(len(samples) * args.test_frac))
    n_val = max(1, int(len(samples) * args.val_frac))
    test_samples = samples[:n_test]
    val_samples = samples[n_test:n_test + n_val]
    train_samples = samples[n_test + n_val:]
    # test_samples is NEVER passed to anything below this line — no
    # DataLoader, no WriteASConcatDataset pool — see this file's own
    # --test-frac help text for why that separation matters.
    excluded_dirs = {trial_dir for trial_dir, _ in val_samples + test_samples}   # both val AND
                                                                                    # test excluded
                                                                                    # from the
                                                                                    # word-concat pool
                                                                                    # below — a
                                                                                    # synthetic
                                                                                    # training example
                                                                                    # built from a
                                                                                    # held-out trial
                                                                                    # would leak it
                                                                                    # into training
    train_samples_by_participant = {
        p: [(d, t) for d, t in s if d not in excluded_dirs] for p, s in samples_by_participant.items()
    }
    print(f'[SETUP] {len(train_samples)} train / {len(val_samples)} val / {len(test_samples)} test trials')

    args.out_dir.mkdir(parents=True, exist_ok=True)
    splits_path = args.out_dir / 'splits.json'
    with open(splits_path, 'w') as f:
        json.dump({
            'text_splits': {
                'train': [[str(d), t] for d, t in train_samples],
                'val': [[str(d), t] for d, t in val_samples],
                'test': [[str(d), t] for d, t in test_samples],
            },
            'classes': args.classes, 'imu_source': args.imu_source, 'finger': args.finger,
            'seed': args.seed,
        }, f, indent=2)
    print(f'[SETUP] saved {splits_path} — evaluate_writeas.py\'s own --splits-json reads this same '
          f'"text_splits" shape, so the test set stays consistent between training and later evaluation')

    n_classes = len(args.classes)
    bos_idx, eos_idx = n_classes, n_classes + 1

    train_ds = WriteASWordDataset(train_samples, args.classes, args.imu_source, args.finger,
                                   augment=args.augment)
    val_ds = WriteASWordDataset(val_samples, args.classes, args.imu_source, args.finger,
                                 augment=False)   # never augmented — see WriteASWordDataset's
                                                    # own docstring on why

    if args.word_concat_per_epoch > 0:
        concat_ds = WriteASConcatDataset(train_samples_by_participant, args.classes, args.imu_source,
                                          args.word_concat_per_epoch, args.finger, seed=args.seed)
        train_ds = torch.utils.data.ConcatDataset([train_ds, concat_ds])
        print(f'[SETUP] +{args.word_concat_per_epoch} synthetic word-concatenation examples per epoch '
              f'(--word-concat-per-epoch)')

    def collate(batch):
        return collate_writeas(batch, n_classes, bos_idx, eos_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    model = WriteASModel(n_classes=n_classes, sample_rate=config_ctc.IMU_RESAMPLE_HZ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)   # paper's own optimizer choice,
                                                                      # default settings (Section 4.3.3)
    ctc_loss_fn = torch.nn.CTCLoss(blank=0, zero_infinity=True)

    best_val_loss = float('inf')
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f'epoch {epoch}/{args.epochs}')
        for imu, imu_len, ctc_targets, ctc_target_len, attn_input, attn_target, attn_len in pbar:
            imu, imu_len = imu.to(device), imu_len.to(device)
            ctc_targets, ctc_target_len = ctc_targets.to(device), ctc_target_len.to(device)
            attn_input, attn_target = attn_input.to(device), attn_target.to(device)
            attn_len = attn_len.to(device)

            optimizer.zero_grad()
            ctc_log_probs, n_clips, attn_logits = model(imu, imu_len, target_tokens=attn_input,
                                                          teacher_forcing=True)
            # attn_input includes <BOS> as step 0 -- the model's own step t output PREDICTS
            # target token t (matching attn_target's own alignment, which has NO <BOS>) -- so
            # attn_logits[:, :-1] (drop the last step, whose input was the true last real
            # letter) aligns 1:1 with attn_target[:, :attn_input.shape[1]-1].
            attn_logits_aligned = attn_logits[:, :attn_target.shape[1], :]
            loss_ctc = ctc_loss_fn(ctc_log_probs, ctc_targets, n_clips, ctc_target_len)
            loss_attn = compute_attention_loss(attn_logits_aligned, attn_target, attn_len)
            loss = (1 - args.ctc_attention_lambda) * loss_attn + args.ctc_attention_lambda * loss_ctc

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix(loss=f'{loss.item():.3f}')

        train_loss = running_loss / max(len(train_loader), 1)

        model.eval()
        val_loss_sum = 0.0
        n_correct_ctc = 0
        n_total = 0
        with torch.no_grad():
            for imu, imu_len, ctc_targets, ctc_target_len, attn_input, attn_target, attn_len in val_loader:
                imu, imu_len = imu.to(device), imu_len.to(device)
                ctc_targets, ctc_target_len = ctc_targets.to(device), ctc_target_len.to(device)
                attn_input, attn_target = attn_input.to(device), attn_target.to(device)
                attn_len = attn_len.to(device)

                ctc_log_probs, n_clips, attn_logits = model(imu, imu_len, target_tokens=attn_input,
                                                              teacher_forcing=True)
                attn_logits_aligned = attn_logits[:, :attn_target.shape[1], :]
                loss_ctc = ctc_loss_fn(ctc_log_probs, ctc_targets, n_clips, ctc_target_len)
                loss_attn = compute_attention_loss(attn_logits_aligned, attn_target, attn_len)
                loss = (1 - args.ctc_attention_lambda) * loss_attn + args.ctc_attention_lambda * loss_ctc
                val_loss_sum += loss.item()

                preds = ctc_greedy_decode_writeas(ctc_log_probs, n_clips, args.classes)
                offset = 0
                for i, tlen in enumerate(ctc_target_len.tolist()):
                    true_letters = ''.join(args.classes[c] for c in
                                            ctc_targets[offset:offset + tlen].tolist())
                    offset += tlen
                    if preds[i] == true_letters:
                        n_correct_ctc += 1
                    n_total += 1
        val_loss = val_loss_sum / max(len(val_loader), 1)
        val_acc = n_correct_ctc / max(n_total, 1)
        print(f'[EPOCH {epoch:3d}/{args.epochs}] train_loss={train_loss:.4f}  '
              f'val_loss={val_loss:.4f}  val_ctc_exact_accuracy={val_acc:.3f}')

        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'classes': args.classes,
                'imu_source': args.imu_source,
                'finger': args.finger,
                'n_classes': n_classes,
                'val_loss': val_loss,
                'val_ctc_exact_accuracy': val_acc,
            }, args.out_dir / 'best_model.pt')

    with open(args.out_dir / 'training_config.json', 'w') as f:
        json.dump({k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}, f, indent=2)
    print(f'[DONE] best val_loss={best_val_loss:.4f} — saved to {args.out_dir}/best_model.pt')


if __name__ == '__main__':
    main()