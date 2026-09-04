"""
train_ctc_distill.py
────────────────────────────────────────────────────────────────────────────
One-command wrapper for the teacher -> student knowledge distillation
workflow (see train_ctc.py's own --distill-teacher-checkpoint, and the
chat this was built in for why real KD needs a SEPARATELY trained
audio-only teacher rather than just an auxiliary reconstruction loss).

Runs train_ctc.py TWICE, as subprocesses (matching this project's own
established orchestration pattern — train_leave_one_out.py,
sweep_audio_features.py — rather than merging everything into
train_ctc.py's own single-process logic):

  1. TEACHER — --modality audio --audio-source surface. Skipped
     entirely if a checkpoint already exists at --teacher-out-dir
     (override with --retrain-teacher to force a fresh run).
  2. STUDENT — whatever --modality/--audio-source/--imu-source you
     actually want (e.g. fusion), with --distill-teacher-checkpoint
     automatically pointed at step 1's own checkpoint.

Every OTHER flag (--participants-root, --classes, --no-space,
--synthetic-per-epoch, --batch-size, --curriculum, ...) is passed to
BOTH runs UNCHANGED — the teacher and student must agree on the exact
same class list (--no-space changes what the classes actually ARE, so
letting them diverge here would silently break distillation — see
train_ctc.py's own --distill-teacher-checkpoint validation, which
already refuses to run if the two checkpoints' saved class lists don't
match; this script's own shared-passthrough design is what GUARANTEES
they always match in the first place, rather than relying on the user
to remember to type --no-space twice identically).

Usage:
    python train_ctc_distill.py \\
        --participants-root ../dataset --classes a b c d e f g h i j k l m n o p q r s t u v w x y z \\
        --no-space \\
        --modality fusion --audio-source watch --imu-source watch \\
        --epochs 100 --synthetic-per-epoch 3000 --concat-letters-min 2 --concat-letters-max 12 \\
        --distill-weight 0.5 --distill-temperature 4.0 \\
        --teacher-out-dir checkpoints/teacher_surface \\
        --out-dir checkpoints/final_fusion_kd

    # Reuse an already-trained teacher (skips step 1 entirely):
    python train_ctc_distill.py ... --teacher-out-dir checkpoints/teacher_surface
        # (no --retrain-teacher -> if checkpoints/teacher_surface/best_model.pt
        #  already exists, step 1 is skipped automatically)
"""

import argparse
import subprocess
import sys
from pathlib import Path


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description='Trains an audio-only teacher (if not already present), then a student with '
                     'real knowledge distillation from it — one command instead of two manual '
                     'train_ctc.py invocations. See this module\'s own docstring for the full design.')
    # STUDENT's own identity — everything else the student needs (--participants-root, --classes,
    # --no-space, --epochs' worth of OTHER hyperparameters, etc.) is passthrough (see parse_known_args
    # below), shared with the teacher run automatically.
    parser.add_argument('--modality', required=True, choices=['fusion', 'audio', 'imu'],
                         help='the STUDENT\'s own real modality — e.g. "fusion" for the actual '
                              'deployed watch-audio + watch-IMU model this whole workflow exists to '
                              'improve. The teacher is ALWAYS --modality audio --audio-source surface '
                              'regardless of what you put here — that\'s fixed, not a passthrough flag.')
    parser.add_argument('--audio-source', default='watch',
                         help='student\'s own audio source (default "watch" — matching the real, '
                              'deployed setup this KD workflow is meant to improve).')
    parser.add_argument('--imu-source', default='watch',
                         help='student\'s own IMU source (only matters for --modality fusion/imu).')
    parser.add_argument('--epochs', type=int, required=True,
                         help='STUDENT\'s own epoch count. See --teacher-epochs for the teacher\'s own '
                              '(independent) epoch count.')
    parser.add_argument('--out-dir', type=Path, required=True,
                         help='STUDENT\'s own final checkpoint directory — this is the one you\'ll '
                              'actually evaluate/deploy from.')
    parser.add_argument('--distill-weight', type=float, default=0.5)
    parser.add_argument('--distill-temperature', type=float, default=4.0)

    # TEACHER-specific controls
    parser.add_argument('--teacher-out-dir', type=Path, required=True,
                         help='where the teacher\'s own checkpoint lives (or already lives, if '
                              'reusing one). --modality/--audio-source are FIXED to audio/surface for '
                              'this run regardless of the student\'s own settings above — see this '
                              'module\'s own docstring.')
    parser.add_argument('--teacher-epochs', type=int, default=None,
                         help='teacher\'s own epoch count — defaults to the SAME value as --epochs '
                              '(the student\'s) if not given separately. A clean audio-only model '
                              'often converges faster than a full fusion model, so a smaller value '
                              'here is a reasonable thing to try independently.')
    parser.add_argument('--retrain-teacher', action='store_true',
                         help='force a fresh teacher training run even if --teacher-out-dir/'
                              'best_model.pt already exists (default: reuse it automatically, '
                              'skipping step 1 entirely, matching this project\'s own established '
                              '--skip-existing-style conventions).')
    return parser


def main():
    args, passthrough = build_arg_parser().parse_known_args()
    # `passthrough` carries every OTHER flag verbatim — --participants-root, --classes, --no-space,
    # --synthetic-per-epoch, --batch-size, --curriculum, --sequence-encoder, --motion-loss-weight,
    # etc. — to BOTH the teacher and student runs below, unchanged. This is what guarantees the two
    # checkpoints' class lists (and every other shared setting) always match — see this module's own
    # top-level docstring.

    train_script = Path(__file__).resolve().parent / 'train_ctc.py'
    teacher_checkpoint = args.teacher_out_dir / 'best_model.pt'

    if args.retrain_teacher or not teacher_checkpoint.exists():
        teacher_epochs = args.teacher_epochs if args.teacher_epochs is not None else args.epochs
        print(f'\n{"=" * 70}\n[STEP 1/2] Training teacher (--modality audio --audio-source surface, '
              f'{teacher_epochs} epochs) -> {args.teacher_out_dir}\n{"=" * 70}\n')
        teacher_cmd = [
            sys.executable, str(train_script),
            '--modality', 'audio', '--audio-source', 'surface',
            '--epochs', str(teacher_epochs),
            '--out-dir', str(args.teacher_out_dir),
        ] + passthrough
        result = subprocess.run(teacher_cmd)
        if result.returncode != 0:
            raise RuntimeError(f'teacher training failed (exit code {result.returncode}) — aborting '
                                f'before starting student training. See the teacher run\'s own output '
                                f'above for details.')
    else:
        print(f'\n[STEP 1/2] Reusing existing teacher checkpoint at {teacher_checkpoint} '
              f'(pass --retrain-teacher to force a fresh run instead)\n')

    print(f'\n{"=" * 70}\n[STEP 2/2] Training student (--modality {args.modality} --audio-source '
          f'{args.audio_source}, distilling from {teacher_checkpoint}) -> {args.out_dir}\n{"=" * 70}\n')
    student_cmd = [
        sys.executable, str(train_script),
        '--modality', args.modality, '--audio-source', args.audio_source,
        '--imu-source', args.imu_source,
        '--epochs', str(args.epochs),
        '--out-dir', str(args.out_dir),
        '--distill-teacher-checkpoint', str(teacher_checkpoint),
        '--distill-weight', str(args.distill_weight),
        '--distill-temperature', str(args.distill_temperature),
    ] + passthrough
    result = subprocess.run(student_cmd)
    if result.returncode != 0:
        raise RuntimeError(f'student training failed (exit code {result.returncode})')

    print(f'\n[DONE] teacher: {teacher_checkpoint}\n[DONE] student (final model): {args.out_dir}/best_model.pt')


if __name__ == '__main__':
    main()
