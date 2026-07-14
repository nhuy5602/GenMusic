# Training and Evaluation

## 1. Prepare Audio Data

Raw WAV/MP3 files are discovered recursively. Demucs separates vocals and
backing, while Whisper supplies the lyric transcript:

```powershell
uv run python cli.py preprocess-raw --input dataset/vietnamese_songs --output dataset/diff_rhythm_dataset --whisper-model small --max-files 100 --keep-separated-count 10
```

If some files fail, the command returns `completed_with_warnings` and a non-zero
exit code. Inspect the printed failure list before training.

## 2. Create a Smoke Dataset

Use synthetic Mel tensors to verify the pipeline without downloading audio:

```powershell
uv run python cli.py make-random-dataset --out dataset/random_self_diffusion_training --count 16 --frames 128 --target-gb 1.0
uv run python cli.py validate-dataset --dataset dataset/random_self_diffusion_training
```

The synthetic dataset validates software behavior only; it is not a substitute
for real singing data.

## 3. Train the Self-authored Model

```powershell
uv run python cli.py train-self --dataset dataset/random_self_diffusion_training --checkpoint outputs/self_music_checkpoint.pt --epochs 2 --batch-size 4
```

For separated vocal/backing records, use the optional MicroDiT path:

```powershell
uv run python cli.py train-self --dataset dataset/diff_rhythm_dataset --checkpoint outputs/microdit.pt --model-type dit --epochs 2 --batch-size 2
```

The MicroDiT path may download its text encoder on first use and therefore
requires network access unless the encoder is already cached.

## 4. Optional Distillation

```powershell
uv run python cli.py train-distill --dataset dataset/diff_rhythm_dataset --student-checkpoint outputs/distilled_student.pt --teacher-checkpoint outputs/teacher.pt --epochs 5 --batch-size 4
```

Provide a local teacher checkpoint for a reproducible run. The fallback teacher
exists for smoke testing and does not improve musical quality by itself.

## 5. Generate and Evaluate

```powershell
uv run python cli.py generate-local --text "Dem nay thanh pho ngu quen trong tieng mua." --style "soft Vietnamese ballad" --duration 8 --checkpoint outputs/self_music_checkpoint.pt --out outputs/demo
uv run python cli.py evaluate-self --generated outputs/demo/final.wav --out outputs/demo/evaluation
uv run python cli.py project-report --source outputs --out outputs/project_report
```

Evaluation writes objective metrics and plots. MOS/CMOS remain skipped unless a
human listening survey is supplied.
