"""
train_ctc.py
────────────────────────────────────────────────────────────────────────────
Trains LetterCTCNet on ordinary single-letter trials — reuses dataset.py's
scan_dataset()/make_splits()/save_splits()/load_splits() UNMODIFIED to find
and split trials (that logic has nothing to do with fixed-vs-variable-
length loading, so there's no reason to duplicate it); only the actual
per-trial audio/IMU loading (dataset_ctc.py) and the model (model_ctc.py)
are new.

Every training sample's target sequence is length 1 (just that one
letter) — CTC doesn't need per-timestep ground truth to learn from single-
segmented letters; the payoff is at INFERENCE time (see
evaluate_word_ctc.py), where the same trained model can be run on a whole
unsegmented word span instead.

Usage:
    python train_ctc.py --classes a b c d e f g h i j k l m n o p q r s t u v w x y z \\
        --modality imu --imu-source fingertip --out-dir checkpoints/letter_ctc_fingertip
"""

import argparse
import json
import sys
from pathlib import Path

# The CTC files live in their own subfolder (e.g. ml/260806test_ctc/) so
# they're easy to keep separate from the original fixed-length pipeline —
# but config.py/dataset.py/model.py (unprefixed, reused as-is) live one
# level up in ml/ itself. Python only auto-adds the SCRIPT's own directory
# to sys.path, never its parent, so without this those imports fail with
# "No module named 'config'" the moment this file isn't run from ml/
# directly. Inserting the parent directory once, here, before any of this
# file's own `import config`/`from dataset import ...` lines, is enough —
# sys.path is process-wide, so config_ctc.py's/dataset_ctc.py's own
# `import config` (triggered when THIS file imports them, below) sees the
# same fixed-up path automatically; it doesn't need repeating in each file.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')   # this runs in a terminal, not a notebook — never try to pop up a GUI window,
                          # just render straight to a file (see plot_history() below)
import matplotlib.pyplot as plt
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False   # --tensorboard just prints a warning and continues without it —
                               # see main()'s handling of this flag

import config
import config_ctc
from dataset import scan_dataset, make_splits, save_splits
from dataset_ctc import LetterDatasetCTC, collate_fn_ctc, discover_participant_dataset_dirs, \
    load_audio_variable, load_imu_variable
from dataset_ctc_concat import ConcatLetterDatasetCTC
from dataset_ctc_realtext import scan_text_trials, split_text_trials, RealTextDatasetCTC, get_word_span
from model_ctc import build_model_ctc, ctc_greedy_decode



def compute_motion_loss(traj_pred: torch.Tensor, traj_padded: torch.Tensor, traj_lengths: torch.Tensor,
                         has_traj: torch.Tensor, imu_lengths: torch.Tensor):
    """MSE between the IMU decoder's reconstructed trajectory (traj_pred,
    at IMU's OWN native time resolution — IMUEncoderTCN never downsamples
    T, see its TIME_DOWNSAMPLE=1) and the real normalized trajectory
    (traj_padded, at config_ctc.IMU_RESAMPLE_HZ — usually a close but not
    always EXACTLY matching resolution, hence the per-sample interpolation
    below), averaged only over samples that actually have real trajectory
    ground truth (has_traj — see dataset_ctc.load_trajectory_variable()'s
    docstring for why some samples legitimately don't). Returns None (not
    zero) if no sample in this batch has trajectory ground truth, so the
    caller can skip adding it to the loss entirely rather than adding a
    real zero — a distinction that matters for anything logging/averaging
    this loss over epochs.

    NOTE: takes imu_lengths (each sample's real, pre-padding IMU length),
    NOT out_lengths (the RNN/fused-features length) — traj_pred lives at
    IMU's own resolution, which is generally NOT the same as the fused
    sequence's downsampled resolution used elsewhere in this file."""
    if not has_traj.any():
        return None
    losses = []
    for i in range(traj_pred.shape[0]):
        if not has_traj[i]:
            continue
        t_len = int(traj_lengths[i])
        o_len = int(imu_lengths[i])
        if t_len < 2 or o_len < 1:
            continue
        pred_i = traj_pred[i, :o_len, :]                              # (T_imu_i, 2)
        gt_i = traj_padded[i, :, :t_len].unsqueeze(0)                   # (1, 2, T_gt_i)
        gt_resized = F.interpolate(gt_i, size=o_len, mode='linear', align_corners=False)
        gt_resized = gt_resized.squeeze(0).transpose(0, 1)               # (T_imu_i, 2)
        losses.append(F.mse_loss(pred_i, gt_resized))
    if not losses:
        return None
    return torch.stack(losses).mean()


def compute_spec_loss(spec_recon: torch.Tensor, surface_target: torch.Tensor, surface_lengths: torch.Tensor,
                       audio_lengths: torch.Tensor):
    """MSE between the Mic decoder's (U-Net's) reconstruction and the
    real SURFACE MIC spectrogram — the reconstruction target is always
    the surface mic recording, regardless of what audio_source the
    model's actual INPUT uses (see dataset_ctc.LetterDatasetCTC's
    surface_audio, loaded independently of the run's own --audio-source)
    — mirroring exactly how the IMU decoder's target is always
    fingertip_imu.csv regardless of --imu-source (see
    compute_motion_loss). When --audio-source is already 'surface' this
    degenerates to plain self-reconstruction (input and target are the
    same recording) — a harmless, expected special case, not a bug.

    spec_recon lives at the INPUT audio's own time resolution
    (audio_lengths); surface_target lives at the surface mic's own
    (usually close but not always identical) time resolution
    (surface_lengths) — interpolated per-sample to reconcile, same idea
    as compute_motion_loss's trajectory resizing."""
    losses = []
    for i in range(spec_recon.shape[0]):
        recon_len = int(audio_lengths[i])
        target_len = int(surface_lengths[i])
        if recon_len < 1 or target_len < 1:
            continue
        recon_i = spec_recon[i:i + 1, :, :, :recon_len]              # (1, 1, N_MELS, recon_len)
        target_i = surface_target[i:i + 1, :, :, :target_len]          # (1, 1, N_MELS, target_len)
        target_resized = F.interpolate(target_i, size=recon_i.shape[2:], mode='bilinear', align_corners=False)
        losses.append(F.mse_loss(recon_i, target_resized))
    if not losses:
        return None
    return torch.stack(losses).mean()


def evaluate(model, loader, classes, device, n_samples_to_show: int = 8):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    n_correct = 0
    n_total = 0
    n_blank_only = 0   # decoded to an empty string — the "blank collapse" phase's signature
    examples = []
    loss_fn = torch.nn.CTCLoss(blank=config_ctc.BLANK_IDX, zero_infinity=True)
    with torch.no_grad():
        for (audio, audio_len, imu, imu_len, targets, target_len, _traj, _traj_len, _has_traj,
             _surface, _surface_len) in loader:
            audio, imu = audio.to(device), imu.to(device)
            audio_len, imu_len = audio_len.to(device), imu_len.to(device)
            targets, target_len = targets.to(device), target_len.to(device)

            log_probs, out_len, _traj_pred, _spec_recon = model(audio, audio_len, imu, imu_len)
            loss = loss_fn(log_probs, targets, out_len, target_len)
            total_loss += loss.item()
            n_batches += 1

            decoded = ctc_greedy_decode(log_probs, out_len, classes)
            # every training/val sample here is a single letter, so the
            # "correct" ground-truth string is always exactly 1 character
            true_chars = [classes[t.item() - 1] for t in targets]   # target_len is always 1 per sample
            n_correct += sum(d == t for d, t in zip(decoded, true_chars))
            n_blank_only += sum(d == '' for d in decoded)
            n_total += len(decoded)
            if len(examples) < n_samples_to_show:
                examples.extend(list(zip(true_chars, decoded))[:n_samples_to_show - len(examples)])
    model.train()
    diagnostics = {'n_blank_only': n_blank_only, 'n_total': n_total, 'examples': examples}
    return total_loss / max(n_batches, 1), n_correct / max(n_total, 1), diagnostics


def edit_distance(a: str, b: str) -> int:
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, dp[0] = dp[0], i
        for j, cb in enumerate(b, 1):
            prev, dp[j] = dp[j], min(dp[j] + 1, dp[j - 1] + 1, prev + (ca != cb))
    return dp[-1]


def evaluate_word_level(model, text_val_samples: list, classes: list, device, modality: str,
                         audio_source: str, imu_source: str, finger: str, n_samples_to_show: int = 5):
    """Runs CTC decode on whole, un-segmented word/sentence trials (same
    approach as evaluate_word_ctc.py — first touch_on to last touch_off,
    no pre-segmentation) and reports word-level accuracy/edit distance.

    This exists because letter-only val_loss (see evaluate() above) is a
    genuinely different thing to measure than word-decoding quality — a
    checkpoint can have excellent single-letter val_loss while never
    having been meaningfully exposed to multi-character targets yet (e.g.
    early in a --curriculum run's stage 1), and picking "best" purely by
    letter val_loss can end up saving exactly that checkpoint: good at
    isolated letters, catastrophically bad at real words (collapses to a
    single character regardless of input length — see the chat this was
    added in for the exact symptom). Whenever real word/sentence val data
    is available (--use-word-trials/--use-sentence-trials), THIS is the
    metric that should decide which checkpoint is "best", not letter
    val_loss alone — see main()'s checkpoint-selection logic."""
    model.eval()
    total_edit_distance = 0
    total_chars = 0
    n_correct = 0
    n_total = 0
    n_targets_with_space = 0    # how many val sentences even HAD a space to learn from
    n_preds_with_space = 0      # how many predictions actually contained one
    examples = []
    with torch.no_grad():
        for trial_dir, true_text in text_val_samples:
            span = get_word_span(trial_dir)
            if span is None:
                continue
            t_start, t_end = span
            audio = audio_len = imu = imu_len = None
            if modality in ('audio', 'fusion'):
                a = load_audio_variable(trial_dir, audio_source, t_start, t_end).unsqueeze(0).to(device)
                audio, audio_len = a, torch.tensor([a.shape[3]]).to(device)
            if modality in ('imu', 'fusion'):
                i = load_imu_variable(trial_dir, imu_source, finger, t_start, t_end).unsqueeze(0).to(device)
                imu, imu_len = i, torch.tensor([i.shape[2]]).to(device)

            log_probs, out_len, _traj_pred, _spec_recon = model(audio, audio_len, imu, imu_len)
            pred = ctc_greedy_decode(log_probs, out_len, classes)[0]

            total_edit_distance += edit_distance(pred, true_text)
            total_chars += len(true_text)
            n_correct += (pred == true_text)
            n_total += 1
            if ' ' in true_text:
                n_targets_with_space += 1
            if ' ' in pred:
                n_preds_with_space += 1
            if len(examples) < n_samples_to_show:
                examples.append((true_text, pred))
    model.train()
    mean_edit_distance = total_edit_distance / max(n_total, 1)
    normalized_edit_distance = total_edit_distance / max(total_chars, 1)   # scale-free — comparable
                                                                             # across checkpoints even if
                                                                             # the val sample mix shifts
    accuracy = n_correct / max(n_total, 1)
    space_stats = {'n_targets_with_space': n_targets_with_space, 'n_preds_with_space': n_preds_with_space,
                   'n_total': n_total}
    return normalized_edit_distance, mean_edit_distance, accuracy, examples, space_stats


def plot_history(history: list, out_path: Path, curriculum_transitions: list = None):
    """Renders train/val loss, val letter accuracy, and (whenever
    computed — see --word-eval-every) word-level normalized edit distance
    / exact accuracy as one multi-panel PNG, overwritten after every
    epoch so progress can be checked mid-run by just opening the file —
    no need to wait for training to finish or parse the console log by
    hand. curriculum_transitions (epoch numbers where --curriculum
    switched stages) are drawn as vertical dashed lines across every
    panel, since a stage change is often exactly what explains a sudden
    jump in these curves (see the chat this was built for — e.g. val_loss
    spiking right as real word/sentence data enters training)."""
    epochs = [h['epoch'] for h in history]
    word_epochs = [h['epoch'] for h in history if h.get('word_normalized_edit_distance') is not None]
    word_ned = [h['word_normalized_edit_distance'] for h in history if h.get('word_normalized_edit_distance') is not None]
    word_acc = [h['word_exact_accuracy'] for h in history if h.get('word_exact_accuracy') is not None]

    has_word = len(word_epochs) > 0
    n_panels = 3 if has_word else 2
    fig, axes = plt.subplots(n_panels, 1, figsize=(9, 3.2 * n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]

    ax = axes[0]
    ax.plot(epochs, [h['train_loss'] for h in history], label='train_loss', color='#1f77b4')
    ax.plot(epochs, [h['val_loss'] for h in history], label='val_loss (letters)', color='#d62728')
    ax.set_ylabel('CTC loss')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title('Training progress')

    ax = axes[1]
    ax.plot(epochs, [h['val_acc'] for h in history], label='val_letter_acc', color='#2ca02c')
    ax.set_ylabel('letter accuracy')
    ax.set_ylim(0, 1)
    ax.legend(loc='upper left', fontsize=8)

    if has_word:
        ax = axes[2]
        ax.plot(word_epochs, word_ned, label='word normalized edit distance (lower=better)',
                 color='#9467bd', marker='o', markersize=3)
        ax.plot(word_epochs, word_acc, label='word exact accuracy', color='#ff7f0e', marker='o', markersize=3)
        ax.set_ylabel('word-level metric')
        ax.set_ylim(0, 1.05)
        ax.legend(loc='upper right', fontsize=8)

    for ax in axes:
        ax.set_xlabel('epoch')
        ax.grid(alpha=0.3)
        if curriculum_transitions:
            for t_epoch, label in curriculum_transitions:
                ax.axvline(t_epoch, color='gray', linestyle='--', alpha=0.6, linewidth=1)
        if ax is axes[0] and curriculum_transitions:
            for t_epoch, label in curriculum_transitions:
                ax.annotate(label, (t_epoch, ax.get_ylim()[1]), fontsize=7, color='gray',
                            rotation=90, va='top', ha='right')

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Train a CTC letter model — see this file\'s docstring')
    parser.add_argument('--participants-root', type=Path,
                         default=Path(__file__).resolve().parent.parent.parent / 'dataset',
                         help='folder containing one subfolder per participant (p1/, p2/, p3/, ...), each '
                              'with its own dataset/<class>/trial_XXX/ inside — e.g. '
                              '<this>/p3/dataset/a/trial_001/. Every "p*"-named folder found here is '
                              'pooled into one training set automatically; adding a new participant folder '
                              'later needs no command-line or code change to be picked up on the next run. '
                              'Default assumes this script still lives 3 folders below it (see this file\'s '
                              'own location) — pass this explicitly if that\'s no longer true.')
    parser.add_argument('--participants', nargs='+', default=None,
                         help='restrict to specific participants (e.g. --participants p1 p3) — default: '
                              'use every participant folder found under --participants-root')
    parser.add_argument('--classes', nargs='+', required=True, help='e.g. a b c d ... z')
    parser.add_argument('--modality', choices=['fusion', 'audio', 'imu'], default='fusion')
    parser.add_argument('--audio-source', choices=list(config.AUDIO_FILENAMES), default=config.DEFAULT_AUDIO_SOURCE)
    parser.add_argument('--imu-source', choices=list(config.IMU_FILENAMES), default=config.DEFAULT_IMU_SOURCE)
    parser.add_argument('--finger', default=config.DEFAULT_FINGER)
    parser.add_argument('--train-frac', type=float, default=0.7)
    parser.add_argument('--val-frac', type=float, default=0.15)
    parser.add_argument('--test-frac', type=float, default=0.15)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--dropout', type=float, default=0.3,
                         help='applied inside the GRU stack and once more before the classifier head — '
                              'higher values fight overfitting harder on a small/imbalanced dataset, at '
                              'some cost to how fast the model fits the training data at all')
    parser.add_argument('--rnn-hidden', type=int, default=256,
                         help='hidden size of the sequence-encoder BiGRU (2 layers, bidirectional) that '
                              'reads the fused/gated audio+IMU latent — matches this project\'s architecture '
                              'diagram (up from an earlier default of 128).')
    parser.add_argument('--motion-loss-weight', type=float, default=0.0,
                         help='weight for the IMU decoder\'s auxiliary trajectory-reconstruction loss — '
                              'reads z_imu (the TCN IMU encoder\'s own latent, BEFORE fusion) and is trained '
                              'to reconstruct the real, NORMALIZED (no absolute coordinates — see '
                              'dataset_ctc.load_trajectory_variable) fingertip trajectory from it: '
                              'total_loss = ctc_loss + motion_loss_weight * mse(reconstructed_trajectory, '
                              'real_trajectory) [+ spec_loss_weight * ... , see --spec-loss-weight]. Inspired '
                              'by Watch Your Mouth\'s use of spatiotemporal mouth geometry as a rich '
                              'intermediate representation the model must route information through before '
                              'decoding to text. 0.0 (default) disables this loss term entirely — z_imu is '
                              'still trained purely by CTC loss backpropagating through it either way; this '
                              'only controls whether it\'s ALSO nudged toward matching real physical motion. '
                              'Try 0.1-0.5 as a starting point if enabling it.')
    parser.add_argument('--spec-loss-weight', type=float, default=0.0,
                         help='weight for the Mic (U-Net) encoder\'s auxiliary spectrogram-reconstruction '
                              'loss — the same idea as --motion-loss-weight, but for audio: the U-Net\'s '
                              'decoder path reconstructs the SURFACE MIC spectrogram, ALWAYS from the '
                              '\'surface\' source regardless of what --audio-source the model\'s actual input '
                              'uses (mirroring how the IMU decoder always targets fingertip_imu.csv '
                              'regardless of --imu-source) — so with --audio-source watch, this pushes '
                              'z_mic toward "what would the cleaner surface mic have sounded like", not '
                              'mere self-reconstruction. When --audio-source is already \'surface\' this '
                              'degenerates to plain self-reconstruction (input and target are the same '
                              'recording) — a harmless, expected special case. 0.0 (default) disables this '
                              'loss term entirely. Try 0.1-0.5 as a starting point if enabling it.')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                         help='L2 penalty on the optimizer — another overfitting-vs-underfitting knob, '
                              'independent of --dropout')
    parser.add_argument('--balanced-sampling', action='store_true', default=True,
                         help='(default on) sample training trials so every letter class is seen roughly '
                              'equally often per epoch, regardless of how many trials it actually has — '
                              'without this, classes with only 3-5 trials (see --classes\' actual counts '
                              'in the [DATA] lines at startup) get proportionally starved during training')
    parser.add_argument('--no-balanced-sampling', dest='balanced_sampling', action='store_false')
    parser.add_argument('--synthetic-per-epoch', type=int, default=0,
                         help='how many synthetic multi-letter "fake word" examples (see '
                              'dataset_ctc_concat.py) to mix into each training epoch, alongside the real '
                              'single-letter trials. 0 (default) disables this entirely — matches the '
                              'original single-letter-only training. Without ANY multi-letter targets '
                              'during training, the model never gets a signal to keep emitting different '
                              'letters over time and collapses to outputting a single character even on '
                              'much longer real word trials at inference — a few thousand tends to be a '
                              'reasonable starting point once real trial counts are in the hundreds.')
    parser.add_argument('--concat-letters-min', type=int, default=2)
    parser.add_argument('--concat-letters-max', type=int, default=5)
    parser.add_argument('--use-word-trials', action='store_true',
                         help='mix in the ACTUALLY COLLECTED word trials (dataset/<participant>/dataset/'
                              'word/trial_XXX/) as additional real, continuous multi-letter training data — '
                              'see dataset_ctc_realtext.py. Ground truth comes straight from each trial\'s '
                              'trial_info.json content field. Only the TRAIN portion of these (split the '
                              'same way letters are, see --train-frac etc.) is used for training; val/test '
                              'portions are saved to splits.json so evaluate_word_ctc.py can be pointed at '
                              'just the held-out ones for a fair number.')
    parser.add_argument('--use-sentence-trials', action='store_true',
                         help='same as --use-word-trials but for sentence/ trials — spaces ARE kept in the '
                              'target (mapped to a dedicated space class, see model_ctc.py\'s classifier '
                              'size — n_classes+2, not +1 — and dataset_ctc_realtext.py\'s space_class_idx), '
                              'so the model learns actual word boundaries, not just a continuous letter run. '
                              'Sentences are the longest, most error-prone-to-align real trials available — '
                              'see --sentence-weight to give them more training signal relative to the much '
                              'more numerous word trials.')
    parser.add_argument('--sentence-weight', type=float, default=3.0,
                         help='multiplies the sampling weight of SENTENCE trials specifically within the '
                              'real word/sentence training pool (word trials keep weight 1.0) — sentences '
                              'are typically far outnumbered by words (e.g. 8-47 sentences vs 41-125 words '
                              'per participant in past runs), so without this, sentence-specific signal '
                              '(especially the space class, which basically only appears in sentence '
                              'targets) gets diluted relative to how much there is to learn. Only has an '
                              'effect when --use-sentence-trials is set; ignored otherwise.')
    parser.add_argument('--word-subfolder', default='word')
    parser.add_argument('--sentence-subfolder', default='sentence')
    parser.add_argument('--curriculum', action='store_true',
                         help='train in stages instead of mixing letters/synthetic/real-word-sentence '
                              'together from epoch 1: stage 1 is letters only, stage 2 adds synthetic '
                              'multi-letter concat (dataset_ctc_concat.py), stage 3 additionally adds real '
                              'word/sentence trials (dataset_ctc_realtext.py, if --use-word-trials/'
                              '--use-sentence-trials is set). Each stage keeps everything the previous stage '
                              'had — nothing is dropped, only added — so the model never un-learns the '
                              'easier task while picking up the harder one. Motivation: a model trained only '
                              'on length-1 targets, then abruptly evaluated on length-13 real words, tends '
                              'to collapse to emitting a single character regardless of how long the input '
                              'actually is (see the chat this was added in) — easing in progressively longer '
                              'targets is the standard curriculum-learning fix for exactly that failure mode.')
    parser.add_argument('--curriculum-stage1-epochs', type=int, default=10,
                         help='epochs 1..N: letters only. IGNORED if --curriculum-stage1-steps is set — see '
                              'that flag\'s help for why epoch-based staging can be unfair when the stages '
                              'have very different dataset sizes.')
    parser.add_argument('--curriculum-stage2-epochs', type=int, default=10,
                         help='epochs N+1..N+M: + synthetic multi-letter concat added. Remaining epochs '
                              '(if any): + real word/sentence trials also added (stage 3). IGNORED if '
                              '--curriculum-stage2-steps is set.')
    parser.add_argument('--curriculum-stage1-steps', type=int, default=None,
                         help='if set, stage boundaries are measured in cumulative GRADIENT STEPS (batches '
                              'trained on so far) instead of epochs, and OVERRIDE --curriculum-stage1-epochs. '
                              'Recommended whenever --synthetic-per-epoch/--use-word-trials are also set: '
                              'stage 1 (letters only) has a MUCH smaller active dataset than stage 3 (letters '
                              '+ synthetic + real text combined), so "N epochs" of one is nowhere near "N '
                              'epochs" of actual training amount as the other — e.g. with 801 letter trials '
                              'and 3000 synthetic + 418 real text added later, stage 3\'s epochs have >5x the '
                              'batches of stage 1\'s, so 10 epochs each is a 5x-uneven comparison, not a fair '
                              'one (see the chat this was added in for the exact numbers this was diagnosed '
                              'from). A reasonable starting point: pick the number of steps N epochs of your '
                              'FULLY MIXED (non-curriculum) run would contain — i.e. using the COMBINED size '
                              '(letters + synthetic-per-epoch + real text), not just the letter count alone: '
                              'stage1_steps = N * ((n_letters + synthetic_per_epoch + n_real_text) // '
                              'batch_size). Using just the letter count here undercounts by however large '
                              'synthetic_per_epoch/n_real_text are, which is exactly the mistake this flag '
                              'exists to avoid making elsewhere.')
    parser.add_argument('--curriculum-stage2-steps', type=int, default=None,
                         help='same idea as --curriculum-stage1-steps, for the stage2->stage3 boundary. '
                              'Defaults to the same value as --curriculum-stage1-steps if that\'s set but '
                              'this isn\'t.')
    parser.add_argument('--word-eval-every', type=int, default=5,
                         help='when real word/sentence val data is available (--use-word-trials/'
                              '--use-sentence-trials), how often (in epochs) to run the more expensive '
                              'whole-trial word-level CTC decode on it (see evaluate_word_level()) — this, '
                              'not letter val_loss, becomes the checkpoint-selection criterion once it\'s '
                              'available, since letter val_loss alone can\'t tell a checkpoint that\'s good '
                              'at isolated letters but never learned to spell multi-character sequences from '
                              'one that actually can (see the chat this was added in). Checkpoints are only '
                              'saved on the epochs this runs — every 5th epoch by default — not every epoch, '
                              'since this is noticeably slower than the letter-only validation.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out-dir', type=Path, default=Path('checkpoints/letter_ctc'))
    parser.add_argument('--plot-every', type=int, default=5,
                         help='how often (in epochs) to (re)write training_curves.png in --out-dir — a '
                             'multi-panel plot of train/val loss, val letter accuracy, and (whenever '
                             'available) word-level normalized edit distance/exact accuracy over the run, '
                             'with vertical lines marking any --curriculum stage transitions. 0 disables '
                             'this entirely. Every epoch would be wasteful; this plus one final write after '
                             'training finishes is enough to check progress by just opening the image, no '
                             'need to wait for the run to finish or parse the console log by hand.')
    parser.add_argument('--tensorboard', action='store_true',
                         help='also log every epoch\'s metrics to TensorBoard (train/val loss, val letter '
                              'accuracy, word-level metrics when computed, plus the fusion gate\'s mean value '
                              'as a rough "audio vs IMU trust" indicator) — unlike training_curves.png (a '
                              'static image, refreshed periodically), TensorBoard updates live in a browser '
                              'while training is still running, and lets you compare multiple runs side by '
                              'side. Logs to --out-dir/tensorboard; view with: '
                              'tensorboard --logdir <out-dir>/tensorboard. Requires the tensorboard package '
                              '(pip install tensorboard) — if it\'s not installed, this prints a warning and '
                              'training continues normally without it, same as any other optional extra here.')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[SETUP] device={device}')

    participant_dirs = discover_participant_dataset_dirs(args.participants_root, args.participants)
    if not participant_dirs:
        raise RuntimeError(f'no participant dataset folders found under {args.participants_root} — '
                            f'expected e.g. {args.participants_root / "p1" / "dataset"}')
    print(f'[SETUP] participants: {[name for name, _ in participant_dirs]}')

    samples = []
    text_samples = []   # (trial_dir, text) pairs from word/sentence trials — see --use-word-trials/
                         # --use-sentence-trials; stays empty (and unused below) if neither is set
    for name, pdir in participant_dirs:
        p_samples = scan_dataset(pdir, args.classes, audio_source=args.audio_source, imu_source=args.imu_source)
        print(f'[SETUP]   {name}: {len(p_samples)} trials')
        samples.extend(p_samples)
        if args.use_word_trials:
            w = scan_text_trials(pdir, args.word_subfolder,
                                  audio_source=args.audio_source, imu_source=args.imu_source)
            print(f'[SETUP]   {name}: {len(w)} word trials')
            text_samples.extend(w)
        if args.use_sentence_trials:
            s = scan_text_trials(pdir, args.sentence_subfolder,
                                  audio_source=args.audio_source, imu_source=args.imu_source)
            print(f'[SETUP]   {name}: {len(s)} sentence trials')
            text_samples.extend(s)
    if not samples:
        raise RuntimeError('no usable trials found across any participant — check --classes/--audio-source/'
                            '--imu-source, or that trial folders actually contain the expected files')
    splits = make_splits(samples, args.train_frac, args.val_frac, args.test_frac, args.seed)

    text_splits = None
    if text_samples:
        text_splits = split_text_trials(text_samples, args.train_frac, args.val_frac, args.test_frac, args.seed)
        print(f'[SETUP] real word/sentence trials: {len(text_splits["train"])} train / '
              f'{len(text_splits["val"])} val / {len(text_splits["test"])} test — only the train portion '
              f'is used below; evaluate_word_ctc.py should be pointed at just the test portion (see '
              f'splits.json) for a number that isn\'t inflated by trials the model was trained on directly')

    args.out_dir.mkdir(parents=True, exist_ok=True)
    splits_path = args.out_dir / 'splits.json'
    save_splits(splits, splits_path, args.classes, args.finger, args.seed,
                args.audio_source, args.imu_source)
    if text_splits is not None:
        # Appended as extra keys rather than routed through save_splits() —
        # that function's payload shape (classes/finger/seed/... + a single
        # `splits` dict of (trial_dir, int_label) pairs) is specific to the
        # letter data it was designed for; text trials carry a STRING
        # ground truth per trial, not an int class label, so they don't
        # fit its schema and are simplest kept as their own top-level key.
        with open(splits_path) as f:
            payload = json.load(f)
        payload['text_splits'] = {
            name: [[str(trial_dir), text] for trial_dir, text in pairs]
            for name, pairs in text_splits.items()
        }
        with open(splits_path, 'w') as f:
            json.dump(payload, f, indent=2)

    print(f'[SETUP] {len(splits["train"])} train / {len(splits["val"])} val / '
          f'{len(splits["test"])} test trials  (classes={args.classes}, modality="{args.modality}")')

    train_ds = LetterDatasetCTC(splits['train'], finger=args.finger,
                                 audio_source=args.audio_source, imu_source=args.imu_source)
    val_ds = LetterDatasetCTC(splits['val'], finger=args.finger,
                               audio_source=args.audio_source, imu_source=args.imu_source)

    # Weight each REAL training trial inversely to how many other trials
    # share its class, so a batch is roughly as likely to contain a class
    # with 3 trials as one with 15 — without this, classes with few
    # trials are seen proportionally rarely each epoch and the model has
    # correspondingly little chance to ever learn them (see the actual
    # per-class trial counts printed by scan_dataset above, e.g. w/g/h/n
    # in the run this was added for). val/test are left at the real,
    # unweighted distribution — this is purely a training-time fix, not
    # something that should change what "accuracy" means.
    class_counts = {}
    for _, label in splits['train']:
        class_counts[label] = class_counts.get(label, 0) + 1
    letter_weights = ([1.0 / class_counts[label] for _, label in splits['train']]
                       if args.balanced_sampling else [1.0] * len(splits['train']))

    # Each of these is built once, up front, and kept/dropped from the
    # active training set per-epoch below (see _build_train_loader) rather
    # than rebuilt — ConcatLetterDatasetCTC already generates a FRESH
    # random combination on every __getitem__ call regardless, so nothing
    # is lost by constructing it only once.
    concat_ds = None
    if args.synthetic_per_epoch > 0:
        concat_ds = ConcatLetterDatasetCTC(
            splits['train'], n_synthetic_per_epoch=args.synthetic_per_epoch,
            n_letters_range=(args.concat_letters_min, args.concat_letters_max),
            finger=args.finger, audio_source=args.audio_source, imu_source=args.imu_source, seed=args.seed)
        print(f'[SETUP] +{len(concat_ds)} synthetic multi-letter examples per epoch '
              f'({args.concat_letters_min}-{args.concat_letters_max} letters each) available '
              f'alongside {len(splits["train"])} real single-letter trials')

    text_train_ds = None
    if text_splits is not None and text_splits['train']:
        # Real, continuously-written word/sentence trials — see
        # dataset_ctc_realtext.py's docstring for why this is a stronger
        # training signal than the synthetic concatenations above (real
        # co-articulation between letters, no artificial silence gaps).
        text_train_ds = RealTextDatasetCTC(
            text_splits['train'], args.classes, finger=args.finger,
            audio_source=args.audio_source, imu_source=args.imu_source)
        # Sentence trials get --sentence-weight (default 3.0), word trials
        # stay at 1.0 — a trial's own parent folder name says which it is
        # (matches the same test evaluate_word_ctc.py uses). Without this,
        # sentences are typically heavily outnumbered by words (see
        # --sentence-weight's help), and the space class in particular
        # appears almost nowhere OUTSIDE sentence targets — under-sampling
        # sentences under-samples the only real source of space-class
        # training signal along with it.
        text_weights = [
            args.sentence_weight if trial_dir.parent.name == args.sentence_subfolder else 1.0
            for trial_dir, _ in text_splits['train']
        ]
        n_sentence_trials = sum(1 for w in text_weights if w == args.sentence_weight)
        print(f'[SETUP] +{len(text_train_ds)} real word/sentence trials available for training '
              f'({n_sentence_trials} sentence, weighted x{args.sentence_weight}; '
              f'{len(text_train_ds) - n_sentence_trials} word, weighted x1.0)')

    def _build_train_loader(epoch: int, global_step: int):
        """Decides which of (letters, synthetic concat, real word/
        sentence) are active for this epoch, and builds a fresh
        DataLoader over exactly that combination. Without --curriculum,
        every available piece is active from epoch 1 (the original,
        non-staged behavior). With --curriculum, phase boundaries use
        cumulative STEP count (global_step, steps completed in prior
        epochs) when --curriculum-stage1-steps is set — see that flag's
        help for why that's fairer than epoch count whenever the stages'
        active datasets are very different sizes — falling back to epoch
        count otherwise. See this function's call site for how "only
        rebuild when the active set actually changes" is decided, since
        building a fresh DataLoader every single epoch even when nothing
        changed would be wasted work."""
        datasets = [train_ds]
        weights = list(letter_weights)
        if args.curriculum and args.curriculum_stage1_steps is not None:
            stage1_boundary = args.curriculum_stage1_steps
            stage2_boundary = stage1_boundary + (args.curriculum_stage2_steps or args.curriculum_stage1_steps)
            use_concat = concat_ds is not None and global_step >= stage1_boundary
            use_text = text_train_ds is not None and global_step >= stage2_boundary
        else:
            use_concat = concat_ds is not None and (not args.curriculum or epoch > args.curriculum_stage1_epochs)
            use_text = text_train_ds is not None and (
                not args.curriculum or epoch > args.curriculum_stage1_epochs + args.curriculum_stage2_epochs)
        if use_concat:
            datasets.append(concat_ds)
            weights += [1.0] * len(concat_ds)
        if use_text:
            datasets.append(text_train_ds)
            weights += text_weights
        combined = torch.utils.data.ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
        sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        loader = DataLoader(combined, batch_size=args.batch_size, sampler=sampler, collate_fn=collate_fn_ctc)
        return loader, use_concat, use_text

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_ctc)

    model = build_model_ctc(args.modality, n_classes=len(args.classes), dropout=args.dropout,
                             rnn_hidden=args.rnn_hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = torch.nn.CTCLoss(blank=config_ctc.BLANK_IDX, zero_infinity=True)

    # Selecting the checkpoint by lowest val_loss rather than highest
    # val_acc — with a val set this small (dozens of trials across 26
    # classes), val_acc jumps around in increments of ~1/len(val) and
    # ties constantly (see the run this was changed for: 0.000/0.023/
    # 0.047/0.070 bouncing with no real trend), while val_loss is a
    # continuous, far less noisy signal of the same underlying thing.
    best_val_loss = float('inf')
    best_word_edit_distance = float('inf')
    history = []
    curriculum_transitions = []   # [(epoch, label), ...] — drawn as vertical lines in plot_history()
    global_step = 0   # cumulative batches trained on so far — see --curriculum-stage1-steps

    tb_writer = None
    if args.tensorboard:
        if HAS_TENSORBOARD:
            tb_dir = args.out_dir / 'tensorboard'
            tb_dir.mkdir(parents=True, exist_ok=True)
            tb_writer = SummaryWriter(log_dir=str(tb_dir))
            print(f'[SETUP] TensorBoard logging to {tb_dir} — view with: tensorboard --logdir {tb_dir}')
        else:
            print('[SETUP] --tensorboard was given but the tensorboard package isn\'t installed '
                  '(pip install tensorboard) — continuing without it.')

    train_loader, active_concat, active_text = _build_train_loader(1, global_step)
    if args.curriculum:
        boundary_desc = (f'{args.curriculum_stage1_steps} steps' if args.curriculum_stage1_steps is not None
                          else f'epochs 1-{args.curriculum_stage1_epochs}')
        print(f'[CURRICULUM] epoch 1: letters only (stage 1, {boundary_desc})')
    for epoch in range(1, args.epochs + 1):
        if args.curriculum:
            # Only rebuilds the loader when a phase boundary is actually
            # crossed — every other epoch reuses the one already built,
            # since rebuilding a DataLoader every epoch even when nothing
            # changed would be wasted work. Uses global_step as it stood
            # at the END of the PREVIOUS epoch (steps within the epoch
            # about to run aren't counted yet) — the transition still only
            # takes effect at an epoch boundary, not mid-epoch, but which
            # epoch that lands on now self-corrects for how many steps
            # each stage's epochs actually contain.
            loader, use_concat, use_text = _build_train_loader(epoch, global_step)
            if (use_concat, use_text) != (active_concat, active_text):
                train_loader, active_concat, active_text = loader, use_concat, use_text
                stage_desc = 'letters'
                if active_concat:
                    stage_desc += ' + synthetic concat'
                if active_text:
                    stage_desc += ' + real word/sentence'
                print(f'[CURRICULUM] epoch {epoch} (step {global_step}): now training on {stage_desc}')
                curriculum_transitions.append((epoch, stage_desc))

        model.train()
        running_loss, running_motion_loss, running_spec_loss = 0.0, 0.0, 0.0
        n_batches, n_motion_batches, n_spec_batches = 0, 0, 0
        pbar = tqdm(train_loader, desc=f'epoch {epoch}/{args.epochs}', leave=False)
        for (audio, audio_len, imu, imu_len, targets, target_len, traj, traj_len, has_traj,
             surface, surface_len) in pbar:
            audio, imu = audio.to(device), imu.to(device)
            audio_len, imu_len = audio_len.to(device), imu_len.to(device)
            targets, target_len = targets.to(device), target_len.to(device)
            traj, traj_len, has_traj = traj.to(device), traj_len.to(device), has_traj.to(device)
            surface, surface_len = surface.to(device), surface_len.to(device)

            optimizer.zero_grad()
            log_probs, out_len, traj_pred, spec_recon = model(audio, audio_len, imu, imu_len)
            loss = loss_fn(log_probs, targets, out_len, target_len)
            if args.motion_loss_weight > 0 and traj_pred is not None:
                motion_loss = compute_motion_loss(traj_pred, traj, traj_len, has_traj, imu_len)
                if motion_loss is not None:
                    loss = loss + args.motion_loss_weight * motion_loss
                    running_motion_loss += motion_loss.item()
                    n_motion_batches += 1
            if args.spec_loss_weight > 0 and spec_recon is not None:
                spec_loss = compute_spec_loss(spec_recon, surface, surface_len, audio_len)
                if spec_loss is not None:
                    loss = loss + args.spec_loss_weight * spec_loss
                    running_spec_loss += spec_loss.item()
                    n_spec_batches += 1
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1
            global_step += 1
            postfix = {'loss': f'{running_loss / n_batches:.3f}'}
            if n_motion_batches > 0:
                postfix['motion'] = f'{running_motion_loss / n_motion_batches:.3f}'
            if n_spec_batches > 0:
                postfix['spec'] = f'{running_spec_loss / n_spec_batches:.3f}'
            pbar.set_postfix(**postfix)

        val_loss, val_acc, diag = evaluate(model, val_loader, args.classes, device)
        train_loss = running_loss / max(n_batches, 1)
        blank_pct = diag['n_blank_only'] / max(diag['n_total'], 1) * 100
        print(f'[EPOCH {epoch:3d}/{args.epochs}] train_loss={train_loss:.4f}  '
              f'val_loss={val_loss:.4f}  val_letter_acc={val_acc:.3f}  '
              f'(blank-only predictions: {blank_pct:.0f}%)')
        print('  sample (true -> predicted): '
              + ', '.join(f'{t}->{d or "(blank)"}' for t, d in diag['examples']))
        history.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss, 'val_acc': val_acc,
                         'word_normalized_edit_distance': None, 'word_exact_accuracy': None})

        if tb_writer is not None:
            tb_writer.add_scalar('loss/train', train_loss, epoch)
            tb_writer.add_scalar('loss/val_letters', val_loss, epoch)
            tb_writer.add_scalar('accuracy/val_letters', val_acc, epoch)
            tb_writer.add_scalar('diagnostics/blank_only_pct', blank_pct, epoch)
            if args.modality == 'fusion' and model.last_gate is not None:
                # Mean gate value across the whole batch/timesteps/channels
                # — a rough, single-number summary of "how much this
                # epoch's model leaned on audio vs IMU overall" (1.0 =
                # fully audio, 0.0 = fully IMU); see model_ctc.py's gate
                # for the actual per-timestep, per-channel detail this
                # collapses down from.
                tb_writer.add_scalar('diagnostics/mean_fusion_gate', model.last_gate.mean().item(), epoch)

        # Whenever real word/sentence val data exists, IT decides which
        # checkpoint is "best" — not letter val_loss alone. See
        # evaluate_word_level()'s docstring for why: letter val_loss can't
        # tell a checkpoint that's good at isolated letters but never
        # learned to spell multi-character sequences from one that
        # actually can, and a run with --curriculum specifically produces
        # exactly that kind of checkpoint early on. Only runs every
        # --word-eval-every epochs (word-level decode is slower than the
        # letter-only check above) — checkpoints are correspondingly only
        # saved on those epochs, not every epoch, when this path is active.
        has_word_val = text_splits is not None and text_splits['val']
        if has_word_val and epoch % args.word_eval_every == 0:
            norm_ed, mean_ed, word_acc, word_examples, space_stats = evaluate_word_level(
                model, text_splits['val'], args.classes, device, args.modality,
                args.audio_source, args.imu_source, args.finger)
            print(f'  [WORD VAL] normalized_edit_distance={norm_ed:.3f}  mean_edit_distance={mean_ed:.2f}  '
                  f'exact_accuracy={word_acc:.3f}')
            print(f'  [WORD VAL] space check: {space_stats["n_targets_with_space"]}/{space_stats["n_total"]} '
                  f'val trials actually contain a space; model predicted one in '
                  f'{space_stats["n_preds_with_space"]}/{space_stats["n_total"]} — if the first number is '
                  f'>0 and the second stays 0 for many epochs, the space class isn\'t being learned yet '
                  f'(check --curriculum stage 3 has actually been reached, and how many epochs it\'s had)')
            print('  [WORD VAL] sample (true -> predicted): '
                  + ', '.join(f'"{t}"->"{d or "(blank)"}"' for t, d in word_examples))
            history[-1]['word_normalized_edit_distance'] = norm_ed
            history[-1]['word_exact_accuracy'] = word_acc
            if tb_writer is not None:
                tb_writer.add_scalar('diagnostics/word_normalized_edit_distance', norm_ed, epoch)
                tb_writer.add_scalar('accuracy/word_exact', word_acc, epoch)
                if space_stats['n_total'] > 0:
                    tb_writer.add_scalar('diagnostics/space_prediction_rate',
                                          space_stats['n_preds_with_space'] / space_stats['n_total'], epoch)
            if norm_ed <= best_word_edit_distance:
                best_word_edit_distance = norm_ed
                args.out_dir.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'classes': args.classes,
                    'modality': args.modality,
                    'rnn_hidden': args.rnn_hidden,
                    'finger': args.finger,
                    'audio_source': args.audio_source,
                    'imu_source': args.imu_source,
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                    'word_val_normalized_edit_distance': norm_ed,
                    'word_val_exact_accuracy': word_acc,
                    'epoch': epoch,
                }, args.out_dir / 'best_model.pt')
                print(f'  [WORD VAL] new best — checkpoint saved (epoch {epoch})')
        elif not has_word_val and val_loss <= best_val_loss:
            best_val_loss = val_loss
            args.out_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'classes': args.classes,
                'modality': args.modality,
                'rnn_hidden': args.rnn_hidden,
                'finger': args.finger,
                'audio_source': args.audio_source,
                'imu_source': args.imu_source,
                'val_acc': val_acc,
                'val_loss': val_loss,
                'epoch': epoch,
            }, args.out_dir / 'best_model.pt')

        if args.plot_every > 0 and epoch % args.plot_every == 0:
            args.out_dir.mkdir(parents=True, exist_ok=True)
            plot_history(history, args.out_dir / 'training_curves.png', curriculum_transitions)

    if args.plot_every > 0:
        # One final write regardless of --plot-every's period, so the
        # image on disk always reflects the LAST epoch actually run, not
        # whichever periodic checkpoint happened to land before it.
        args.out_dir.mkdir(parents=True, exist_ok=True)
        plot_history(history, args.out_dir / 'training_curves.png', curriculum_transitions)
        print(f'[PLOT] training curves saved to {args.out_dir / "training_curves.png"}')
    with open(args.out_dir / 'history.json', 'w') as f:
        json.dump({'history': history, 'curriculum_transitions': curriculum_transitions}, f, indent=2)
    if tb_writer is not None:
        tb_writer.close()

    if text_splits is not None and text_splits['val']:
        print(f'[DONE] best word-level normalized edit distance={best_word_edit_distance:.4f} — '
              f'saved to {args.out_dir / "best_model.pt"}')
    else:
        print(f'[DONE] best val_loss={best_val_loss:.4f} — saved to {args.out_dir / "best_model.pt"}')


if __name__ == '__main__':
    main()
