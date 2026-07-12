import os
import sys
import subprocess
from pathlib import Path

def main():
    # Configure UTF-8 console output for Windows
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    # Resolved parent because it is located inside the scripts/ directory
    project_root = Path(__file__).resolve().parents[1]
    sys.path.append(str(project_root))
    cli_path = str(project_root / "cli.py")
    
    # Load paths dynamically from environment
    from src.integrations.kaggle_auto import load_kaggle_api_tokens
    tokens = load_kaggle_api_tokens()
    raw_input = tokens.get("RAW_AUDIO_INPUT_DIR", "dataset/vietnamese_songs")
    processed_output = tokens.get("PROCESSED_DATASET_DIR", "dataset/diff_rhythm_dataset")
    checkpoint_file = tokens.get("MODEL_CHECKPOINT_PATH", "outputs/my_trained_model.pt")
    checkpoint_path = project_root / checkpoint_file
    
    print("======================================================================")
    # Step 1: Preprocess raw audio tracks
    print("🚀 STEP 1: Running Audio Preprocessing Pipeline...")
    preprocess_cmd = [
        sys.executable, cli_path, "preprocess-raw",
        "--input", raw_input,
        "--output", processed_output,
        "--whisper-model", "tiny"
    ]
    print(f"Running: {' '.join(preprocess_cmd)}")
    subprocess.run(preprocess_cmd, cwd=project_root, check=True)
    
    print("\n======================================================================")
    # Step 2: Train the model
    print("🚀 STEP 2: Training the Simplified Music Self-Diffusion Model...")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    train_cmd = [
        sys.executable, cli_path, "train-self",
        "--dataset", processed_output,
        "--checkpoint", str(checkpoint_path),
        "--epochs", "2",
        "--batch-size", "2",
        "--learning-rate", "1e-3"
    ]
    print(f"Running: {' '.join(train_cmd)}")
    subprocess.run(train_cmd, cwd=project_root, check=True)

    print("\n======================================================================")
    # Step 3: Run local generation (Testing)
    print("🚀 STEP 3: Generating Music Track (Local Testing)...")
    output_audio_dir = project_root / "outputs" / "local_test_generation"
    
    generate_cmd = [
        sys.executable, cli_path, "generate-local",
        "--text", "Lòng em nghe tình yêu vỡ tan. Người đó mình hẹn hò, giờ đây anh xa mãi.",
        "--style", "Vietnamese pop, soft piano, emotional melody",
        "--duration", "4.0",
        "--checkpoint", str(checkpoint_path),
        "--steps", "4",
        "--out", str(output_audio_dir)
    ]
    print(f"Running: {' '.join(generate_cmd)}")
    subprocess.run(generate_cmd, cwd=project_root, check=True)
    
    print("\n======================================================================")
    # Step 4: Run evaluation metrics
    print("🚀 STEP 4: Evaluating the Generated Audio Track...")
    eval_dir = project_root / "outputs" / "evaluation_report"
    
    eval_cmd = [
        sys.executable, cli_path, "evaluate-self",
        "--generated", str(output_audio_dir / "final.wav"),
        "--out", str(eval_dir)
    ]
    print(f"Running: {' '.join(eval_cmd)}")
    subprocess.run(eval_cmd, cwd=project_root, check=True)

    print("\n======================================================================")
    print("🎉 Pipeline Completed Successfully!")
    print(f"  - Dataset: dataset/diff_rhythm_dataset/")
    print(f"  - Checkpoint: outputs/my_rhythm_model.pt")
    print(f"  - Generated Audio: {output_audio_dir}/final.wav")
    print(f"  - Evaluation: {eval_dir}/metric_report.json")
    print("======================================================================")

if __name__ == "__main__":
    main()
