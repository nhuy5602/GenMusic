import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Add project root to sys.path to allow imports from src package
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.integrations.kaggle_auto import (
    kaggle_auth_available,
    kaggle_auth_environment,
    kaggle_cli_command,
    load_kaggle_api_tokens,
    resolve_kaggle_username,
    write_source_zip,
)


def _kernel_script_content(
    dataset_slug: str,
    epochs: str = "5",
    batch_size: str = "4",
    learning_rate: str = "0.0002",
    ema_decay: str = "0.999",
    lambda_vocal: str = "1.0",
    save_every_epoch: bool = True,
    max_records: str | None = None,
    dim: str | None = None,
    depth: str | None = None,
    heads: str | None = None,
    open_vocabulary_conditioning: bool = False,
    lexical_holdout_fraction: str = "0.0",
    minimum_lexical_sensitivity: str = "0.05",
    lyric_semantic_weight: str = "0.25",
    lyric_denoised_semantic_weight: str = "0.0",
    lyric_semantic_temperature: str = "0.08",
    minimum_lyric_semantic_accuracy: str = "0.05",
    minimum_lyric_denoised_semantic_accuracy: str = "0.0",
    lyric_unit_semantic_weight: str = "0.25",
    minimum_lyric_unit_accuracy: str = "0.10",
    lyric_unit_denoised_semantic_weight: str = "0.0",
    minimum_lyric_denoised_unit_accuracy: str = "0.0",
    self_rollout_consistency_weight: str = "0.0",
    self_rollout_consistency_probability: str = "0.0",
    self_rollout_step_size: str = "0.125",
    self_rollout_solver_steps: str = "0",
    early_timestep_fraction: str = "0.0",
    early_timestep_max: str = "0.35",
    semantic_pretrain_only: bool = False,
    minimum_epochs: str = "8",
    early_stopping_patience: str = "4",
    reset_optimizer: bool = False,
    reset_ema: bool = False,
    frames_per_chunk: str | None = None,
    resume: bool = False,
    checkpoint_every_steps: str | None = None,
    evaluation_only: bool = False,
    evaluation_raw: bool = False,
    evaluation_steps: str = "32",
    evaluation_guidance_scale: str = "1.5",
    evaluation_solver: str = "euler",
) -> str:
    # This script will run on the Kaggle GPU instance and log errors to output files instead of crashing
    # We use pure ASCII characters to prevent Windows cp1252 encoding crashes during kaggle push
    return f'''import os
import shutil
import subprocess
import sys
import zipfile
import traceback
from pathlib import Path

# Write directory structure for debugging
try:
    input_dir = Path("/kaggle/input")
    structure = []
    for root, dirs, files in os.walk(str(input_dir)):
        structure.append(f"Folder: {{root}}\\n  Dirs: {{dirs}}\\n  Files: {{files[:20]}}\\n")
    Path("/kaggle/working/dir_structure.txt").write_text("\\n".join(structure), encoding="utf-8")
except Exception as de:
    Path("/kaggle/working/dir_error.txt").write_text(str(de), encoding="utf-8")

try:
    print("--- STEP 1: Locating input datasets ---")
    input_dir = Path("/kaggle/input")
    processed_dataset = next(input_dir.rglob("records.jsonl"), None)
    if not processed_dataset or not processed_dataset.is_file():
        raise RuntimeError("Could not find processed dataset records.jsonl in /kaggle/input.")
    processed_dataset = processed_dataset.parent
    print(f"Using processed dataset: {{processed_dataset.resolve()}}")

    print("--- STEP 2: Setting up source code ---")
    # Locate source code either as a mounted directory or as the source zip.
    source_dataset_dir = next(
        (d for d in input_dir.rglob("*") if d.is_dir() and "genmusic-source-" in d.name.lower()), 
        None
    )
    source_root = Path("/kaggle/working/GenMusic")
    source_zip = next(input_dir.rglob("genmusic_vn_source.zip"), None)
    if source_zip and source_zip.is_file():
        source_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(source_zip) as archive:
            archive.extractall(source_root)
    elif source_dataset_dir:
        shutil.copytree(source_dataset_dir, source_root, dirs_exist_ok=True)
    else:
        raise RuntimeError("Could not find the source code dataset directory or zip.")

    print("--- STEP 3: Creating the uv runtime ---")
    uv_executable = shutil.which("uv")
    if not uv_executable:
        # Kaggle currently provides uv, but retain a bounded bootstrap for
        # older images. All project dependency management after this point is
        # performed by uv.
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "uv"],
            check=True,
        )
        uv_executable = shutil.which("uv")
    if not uv_executable:
        raise RuntimeError("uv is unavailable after bootstrap.")

    uv_venv = Path("/kaggle/working/.venv")
    subprocess.run(
        [
            uv_executable,
            "venv",
            "--system-site-packages",
            "--python",
            sys.executable,
            str(uv_venv),
        ],
        check=True,
    )
    uv_python = uv_venv / "bin" / "python"
    subprocess.run(
        [
            uv_executable,
            "pip",
            "install",
            "--python",
            str(uv_python),
            "librosa",
            "imageio-ffmpeg",
            "text2phonemesequence",
            "jiwer",
        ],
        check=True,
    )
    uv_run_python = [
        uv_executable,
        "run",
        "--no-project",
        "--python",
        str(uv_python),
        "python",
    ]

    print("--- STEP 4: Checking CUDA compatibility ---")
    torch_probe = subprocess.run(
        uv_run_python
        + [
            "-c",
            "import torch; print('torch=%s cuda=%s available=%s' % (torch.__version__, torch.version.cuda, torch.cuda.is_available())); print(torch.randn((2, 2), device='cuda') @ torch.randn((2, 2), device='cuda')) if torch.cuda.is_available() else None",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    torch_probe_output = (torch_probe.stdout or "") + chr(10) + (torch_probe.stderr or "")
    print(torch_probe_output, flush=True)
    if torch_probe.returncode != 0:
        print("CUDA smoke test failed; installing P100-compatible Torch.", flush=True)
        subprocess.run(
            [
                uv_executable,
                "pip",
                "install",
                "--python",
                str(uv_python),
                "--no-cache",
                "--force-reinstall",
                "--extra-index-url",
                "https://download.pytorch.org/whl/cu121",
                "torch==2.5.1+cu121",
                "torchaudio==2.5.1+cu121",
            ],
            check=True,
        )
        repaired_probe = subprocess.run(
            uv_run_python
            + [
                "-c",
                "import torch; print('torch=%s cuda=%s available=%s' % (torch.__version__, torch.version.cuda, torch.cuda.is_available())); print(torch.randn((2, 2), device='cuda') @ torch.randn((2, 2), device='cuda')) if torch.cuda.is_available() else None",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        repaired_output = (repaired_probe.stdout or "") + chr(10) + (repaired_probe.stderr or "")
        print(repaired_output, flush=True)
        if repaired_probe.returncode != 0 or "available=True" not in repaired_output:
            raise RuntimeError("CUDA is unavailable after P100 Torch repair.")

    # Add source code to path
    os.environ["PYTHONPATH"] = str(source_root) + os.pathsep + os.environ.get("PYTHONPATH", "")

    if {evaluation_only}:
        print("--- STEP 5: Evaluating novel Vietnamese prompts ---")
        evaluation_checkpoint = next(
            input_dir.rglob("my_trained_model.best.pt"),
            None,
        )
        if not evaluation_checkpoint:
            evaluation_checkpoint = next(
                input_dir.rglob("my_trained_model.pt"),
                None,
            )
        if not evaluation_checkpoint:
            raise RuntimeError(
                "Evaluation requires a best or latest training checkpoint."
            )
        evaluation_output = Path(
            "/kaggle/working/open_vocabulary_evaluation"
        )
        evaluation_command = uv_run_python + [
            str(
                source_root
                / "scripts"
                / "evaluate_open_vocabulary_diffusion.py"
            ),
            "--checkpoint",
            str(evaluation_checkpoint),
            "--output-root",
            str(evaluation_output),
            "--duration",
            "16",
            "--steps",
            "{evaluation_steps}",
            "--guidance-scale",
            "{evaluation_guidance_scale}",
            "--solver",
            "{evaluation_solver}",
            "--device",
            "cuda",
        ]
        evaluation_report = next(
            input_dir.rglob("training_report.json"),
            None,
        )
        if evaluation_report:
            evaluation_command.extend(
                ["--training-report", str(evaluation_report)]
            )
        if {evaluation_raw}:
            evaluation_command.append("--raw-weights")
        subprocess.run(evaluation_command, check=True)
        Path("/kaggle/working/success.txt").write_text(
            "evaluation_success",
            encoding="utf-8",
        )
        print(
            f"Evaluation saved to: {{evaluation_output.resolve()}}"
        )
        raise SystemExit(0)

    print("--- STEP 5: Training model on ALL processed records ---")
    checkpoint_path = Path("/kaggle/working/my_trained_model.pt")
    if {resume}:
        resume_checkpoint = next(
            input_dir.rglob("my_trained_model.pt"),
            None,
        )
        if not resume_checkpoint:
            raise RuntimeError(
                "Resume was requested but no my_trained_model.pt was found."
            )
        shutil.copy2(resume_checkpoint, checkpoint_path)
        print(f"Resuming checkpoint: {{resume_checkpoint}}")

    train_command = uv_run_python + [
        str(source_root / "cli.py"), "train-self",
        "--dataset", str(processed_dataset),
        "--checkpoint", str(checkpoint_path),
        "--epochs", "{epochs}",
        "--batch-size", "{batch_size}",
        "--learning-rate", "{learning_rate}",
        "--ema-decay", "{ema_decay}",
        "--minimum-epochs", "{minimum_epochs}",
        "--early-stopping-patience", "{early_stopping_patience}",
        "--lambda-vocal", "{lambda_vocal}",
        "--device", "cuda",
    ]
    if {max_records is not None}:
        train_command.extend(["--max-records", "{max_records}"])
    if {dim is not None}:
        train_command.extend(["--dim", "{dim}"])
    if {depth is not None}:
        train_command.extend(["--depth", "{depth}"])
    if {heads is not None}:
        train_command.extend(["--heads", "{heads}"])
    if {frames_per_chunk is not None}:
        train_command.extend(
            ["--frames-per-chunk", "{frames_per_chunk}"]
        )
    if {checkpoint_every_steps is not None}:
        train_command.extend(
            ["--checkpoint-every-steps", "{checkpoint_every_steps}"]
        )
    if {open_vocabulary_conditioning}:
        train_command.extend([
            "--open-vocabulary-conditioning",
            "--lexical-holdout-fraction", "{lexical_holdout_fraction}",
            "--minimum-lexical-sensitivity", "{minimum_lexical_sensitivity}",
            "--lyric-semantic-weight", "{lyric_semantic_weight}",
            "--lyric-denoised-semantic-weight", "{lyric_denoised_semantic_weight}",
            "--lyric-semantic-temperature", "{lyric_semantic_temperature}",
            "--minimum-lyric-semantic-accuracy", "{minimum_lyric_semantic_accuracy}",
            "--minimum-lyric-denoised-semantic-accuracy", "{minimum_lyric_denoised_semantic_accuracy}",
            "--lyric-unit-semantic-weight", "{lyric_unit_semantic_weight}",
            "--minimum-lyric-unit-accuracy", "{minimum_lyric_unit_accuracy}",
            "--lyric-unit-denoised-semantic-weight", "{lyric_unit_denoised_semantic_weight}",
            "--minimum-lyric-denoised-unit-accuracy", "{minimum_lyric_denoised_unit_accuracy}",
            "--self-rollout-consistency-weight", "{self_rollout_consistency_weight}",
            "--self-rollout-consistency-probability", "{self_rollout_consistency_probability}",
            "--self-rollout-step-size", "{self_rollout_step_size}",
            "--self-rollout-solver-steps", "{self_rollout_solver_steps}",
            "--early-timestep-fraction", "{early_timestep_fraction}",
            "--early-timestep-max", "{early_timestep_max}",
        ])
    if {resume}:
        train_command.append("--resume")
    if {semantic_pretrain_only}:
        train_command.append("--semantic-pretrain-only")
    if {reset_optimizer}:
        train_command.append("--reset-optimizer")
    if {reset_ema}:
        train_command.append("--reset-ema")
    if {save_every_epoch}:
        # A validation-gated best checkpoint + early stopping run can span many
        # epochs; persist raw weights/optimizer/EMA after every epoch so a
        # session timeout does not lose all progress (see docs/guides on Kaggle
        # preemption).
        train_command.append("--save-every-epoch")
    subprocess.run(train_command, check=True)

    print("--- PIPELINE COMPLETED SUCCESSFULLY ---")
    print(f"Model saved to: {{checkpoint_path.resolve()}}")
    report_path = Path("/kaggle/working/training_report.json")
    if report_path.is_file():
        print("--- training_report.json ---")
        print(report_path.read_text(encoding="utf-8"))
    Path("/kaggle/working/success.txt").write_text("success", encoding="utf-8")

except Exception as e:
    tb = traceback.format_exc()
    print("Error occurred during training:")
    print(tb)
    Path("/kaggle/working/error.txt").write_text(tb, encoding="utf-8")
'''

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--processed-kernel-ref", default=None)
    parser.add_argument(
        "--processed-dataset-ref",
        default=None,
        help=(
            "Explicit Kaggle Dataset containing config.json, records.jsonl, "
            "and latent tensors. Mutually exclusive with "
            "--processed-kernel-ref."
        ),
    )
    parser.add_argument("--lambda-vocal", type=float, default=1.0, help="Weight of auxiliary vocal-only prediction loss (Mixed Pro style, 0.0 disables it).")
    parser.add_argument("--max-records", type=int, default=None, help="Limit training to the first N usable records (for cheap smoke tests).")
    parser.add_argument("--dim", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--heads", type=int, default=None)
    parser.add_argument("--frames-per-chunk", type=int, default=None)
    parser.add_argument("--checkpoint-every-steps", type=int, default=50)
    parser.add_argument("--open-vocabulary-conditioning", action="store_true")
    parser.add_argument("--lexical-holdout-fraction", type=float, default=0.08)
    parser.add_argument("--minimum-lexical-sensitivity", type=float, default=0.05)
    parser.add_argument("--lyric-semantic-weight", type=float, default=0.25)
    parser.add_argument("--lyric-denoised-semantic-weight", type=float, default=0.0)
    parser.add_argument("--lyric-semantic-temperature", type=float, default=0.08)
    parser.add_argument("--minimum-lyric-semantic-accuracy", type=float, default=0.05)
    parser.add_argument(
        "--minimum-lyric-denoised-semantic-accuracy",
        type=float,
        default=0.0,
    )
    parser.add_argument("--lyric-unit-semantic-weight", type=float, default=0.25)
    parser.add_argument("--minimum-lyric-unit-accuracy", type=float, default=0.10)
    parser.add_argument("--lyric-unit-denoised-semantic-weight", type=float, default=0.0)
    parser.add_argument("--minimum-lyric-denoised-unit-accuracy", type=float, default=0.0)
    parser.add_argument("--self-rollout-consistency-weight", type=float, default=0.0)
    parser.add_argument("--self-rollout-consistency-probability", type=float, default=0.0)
    parser.add_argument("--self-rollout-step-size", type=float, default=0.125)
    parser.add_argument("--self-rollout-solver-steps", type=int, default=0)
    parser.add_argument("--early-timestep-fraction", type=float, default=0.0)
    parser.add_argument("--early-timestep-max", type=float, default=0.35)
    parser.add_argument("--semantic-pretrain-only", action="store_true")
    parser.add_argument("--minimum-epochs", type=int, default=8)
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    parser.add_argument("--reset-optimizer", action="store_true")
    parser.add_argument("--reset-ema", action="store_true")
    parser.add_argument(
        "--resume-kernel-ref",
        default=None,
        help="Optional prior training kernel whose my_trained_model.pt is resumed.",
    )
    parser.add_argument(
        "--submit-only",
        action="store_true",
        help="Submit the remote GPU job, persist local state, and return.",
    )
    parser.add_argument(
        "--evaluation-only",
        action="store_true",
        help="Skip training and evaluate a checkpoint kernel on novel text.",
    )
    parser.add_argument(
        "--evaluation-raw",
        action="store_true",
        help="Use raw instead of EMA checkpoint weights during evaluation.",
    )
    parser.add_argument(
        "--evaluation-steps",
        type=int,
        default=32,
        help="Euler steps used by an evaluation-only generation job.",
    )
    parser.add_argument(
        "--evaluation-guidance-scale",
        type=float,
        default=1.5,
        help="Classifier-free guidance scale for evaluation-only generation.",
    )
    parser.add_argument(
        "--evaluation-solver",
        choices=("euler", "heun", "midpoint"),
        default="euler",
        help="ODE solver used by an evaluation-only generation job.",
    )
    args = parser.parse_args()
    if args.evaluation_only and not args.resume_kernel_ref:
        parser.error("--evaluation-only requires --resume-kernel-ref")
    if args.evaluation_raw and not args.evaluation_only:
        parser.error("--evaluation-raw requires --evaluation-only")
    if args.processed_kernel_ref and args.processed_dataset_ref:
        parser.error(
            "--processed-kernel-ref and --processed-dataset-ref "
            "are mutually exclusive"
        )

    # Resolved parent because it is located inside the scripts/ directory
    project_root = Path(__file__).resolve().parents[1]
    tokens = kaggle_auth_environment(load_kaggle_api_tokens())
    username = resolve_kaggle_username(tokens.get("KAGGLE_USERNAME"))
    cli = kaggle_cli_command()

    if not username or not kaggle_auth_available(tokens) or not cli:
        print("❌ Error: Set KAGGLE_USERNAME and KAGGLE_API_TOKEN=KGAT_... (or legacy KAGGLE_KEY) in .env.")
        return

    kaggle_env = {
        **os.environ,
        **tokens,
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
    }
    
    # Credentials are passed in-memory so project .env remains authoritative.

    # KAGGLE_PROCESSED_KERNEL_REF: output of a preprocess kernel run with the fixed
    # run_kaggle_preprocess_all.py (attached via kernel_sources, no credentials needed).
    # KAGGLE_PROCESSED_DATASET_REF: a pre-existing published Dataset (attached via
    # dataset_sources instead) -- kept for compatibility with datasets published before
    # this fix, or shared manually outside this project's scripts.
    processed_kernel_ref = args.processed_kernel_ref or os.getenv("KAGGLE_PROCESSED_KERNEL_REF") or tokens.get("KAGGLE_PROCESSED_KERNEL_REF")
    processed_dataset_ref = args.processed_dataset_ref or os.getenv(
        "KAGGLE_PROCESSED_DATASET_REF"
    ) or tokens.get(
        "KAGGLE_PROCESSED_DATASET_REF"
    )
    if not processed_kernel_ref and not processed_dataset_ref:
        raise RuntimeError(
            "Set --processed-kernel-ref or --processed-dataset-ref; "
            "the repository intentionally has no account-specific fallback."
        )
    processed_dataset_slug = (processed_kernel_ref or processed_dataset_ref).split("/")[-1]
    epochs = args.epochs
    batch_size = args.batch_size
    
    run_id = f"train-run-{int(time.time())}"
    job_dir = project_root / "outputs" / "kaggle_training" / run_id
    dataset_dir = job_dir / "dataset"
    kernel_dir = job_dir / "kernel"
    download_dir = job_dir / "downloaded_model"
    
    for d in (dataset_dir, kernel_dir, download_dir):
        d.mkdir(parents=True, exist_ok=True)

    print("======================================================================")
    print(f"🚀 Initializing Kaggle Job: {run_id}")
    print(f"   Processed data source: {processed_kernel_ref or processed_dataset_ref} ({'kernel' if processed_kernel_ref else 'dataset'})")
    print(
        "   Training config: "
        f"epochs={epochs}, batch_size={batch_size}, "
        f"frames_per_chunk={args.frames_per_chunk}, "
        f"open_vocabulary={args.open_vocabulary_conditioning}"
    )
    print("======================================================================")

    # 1. Zip source code
    print("📦 Zipping local source code...")
    write_source_zip(project_root, dataset_dir / "genmusic_vn_source.zip")

    # 2. Upload source code zip as a Kaggle Dataset
    source_dataset_slug = f"genmusic-source-{run_id}"
    source_dataset_ref = f"{username}/{source_dataset_slug}"
    
    (dataset_dir / "dataset-metadata.json").write_text(json.dumps({
        "title": f"GenMusic Source {run_id}",
        "id": source_dataset_ref,
        "licenses": [{"name": "other"}]
    }, indent=2))

    print(f"📤 Uploading source code to Kaggle Dataset '{source_dataset_ref}'...")
    subprocess.run(cli + ["datasets", "create", "-p", str(dataset_dir), "-r", "zip"], env=kaggle_env, check=True)

    # Wait until dataset is ready on Kaggle
    print("⏳ Waiting for source dataset to be ready...")
    for _ in range(60):
        res = subprocess.run(cli + ["datasets", "status", source_dataset_ref], env=kaggle_env, capture_output=True, text=True)
        if "ready" in res.stdout.lower():
            break
        time.sleep(5)

    # 3. Create Kernel script and metadata (Keep slug/title short to fit Kaggle's 50-char limit)
    kernel_slug = f"genmusic-train-{int(time.time())}"
    kernel_ref = f"{username}/{kernel_slug}"
    
    (kernel_dir / "run_training.py").write_text(
        _kernel_script_content(
            processed_dataset_slug,
            epochs,
            batch_size,
            str(args.learning_rate),
            str(args.ema_decay),
            str(args.lambda_vocal),
            max_records=str(args.max_records) if args.max_records is not None else None,
            dim=str(args.dim) if args.dim is not None else None,
            depth=str(args.depth) if args.depth is not None else None,
            heads=str(args.heads) if args.heads is not None else None,
            open_vocabulary_conditioning=args.open_vocabulary_conditioning,
            lexical_holdout_fraction=str(args.lexical_holdout_fraction),
            minimum_lexical_sensitivity=str(
                args.minimum_lexical_sensitivity
            ),
            lyric_semantic_weight=str(args.lyric_semantic_weight),
            lyric_denoised_semantic_weight=str(
                args.lyric_denoised_semantic_weight
            ),
            lyric_semantic_temperature=str(
                args.lyric_semantic_temperature
            ),
            minimum_lyric_semantic_accuracy=str(
                args.minimum_lyric_semantic_accuracy
            ),
            minimum_lyric_denoised_semantic_accuracy=str(
                args.minimum_lyric_denoised_semantic_accuracy
            ),
            lyric_unit_semantic_weight=str(
                args.lyric_unit_semantic_weight
            ),
            minimum_lyric_unit_accuracy=str(
                args.minimum_lyric_unit_accuracy
            ),
            lyric_unit_denoised_semantic_weight=str(
                args.lyric_unit_denoised_semantic_weight
            ),
            minimum_lyric_denoised_unit_accuracy=str(
                args.minimum_lyric_denoised_unit_accuracy
            ),
            self_rollout_consistency_weight=str(
                args.self_rollout_consistency_weight
            ),
            self_rollout_consistency_probability=str(
                args.self_rollout_consistency_probability
            ),
            self_rollout_step_size=str(args.self_rollout_step_size),
            self_rollout_solver_steps=str(args.self_rollout_solver_steps),
            early_timestep_fraction=str(
                args.early_timestep_fraction
            ),
            early_timestep_max=str(args.early_timestep_max),
            semantic_pretrain_only=args.semantic_pretrain_only,
            minimum_epochs=str(args.minimum_epochs),
            early_stopping_patience=str(
                args.early_stopping_patience
            ),
            reset_optimizer=args.reset_optimizer,
            reset_ema=args.reset_ema,
            frames_per_chunk=(
                str(args.frames_per_chunk)
                if args.frames_per_chunk is not None
                else None
            ),
            resume=bool(args.resume_kernel_ref),
            checkpoint_every_steps=(
                str(args.checkpoint_every_steps)
                if args.checkpoint_every_steps > 0
                else None
            ),
            evaluation_only=args.evaluation_only,
            evaluation_raw=args.evaluation_raw,
            evaluation_steps=str(args.evaluation_steps),
            evaluation_guidance_scale=str(
                args.evaluation_guidance_scale
            ),
            evaluation_solver=str(args.evaluation_solver),
        ),
        encoding="utf-8",
    )
    kernel_sources = [
        source
        for source in (
            processed_kernel_ref,
            args.resume_kernel_ref,
        )
        if source
    ]
    kernel_sources = list(dict.fromkeys(kernel_sources))
    (kernel_dir / "kernel-metadata.json").write_text(json.dumps({
        "id": kernel_ref,
        "title": kernel_slug,
        "code_file": "run_training.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",       # Enable GPU training
        "enable_internet": "true",
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [source_dataset_ref] + ([] if processed_kernel_ref else [processed_dataset_ref]),
        "kernel_sources": kernel_sources,
    }, indent=2))

    # 4. Push Kernel to Kaggle
    print(f"🚀 Pushing training Kernel '{kernel_ref}' to Kaggle (GPU: T4)...")
    subprocess.run(cli + ["kernels", "push", "-p", str(kernel_dir)], env=kaggle_env, check=True)
    state_path = job_dir / "job_state.json"
    state_path.write_text(
        json.dumps(
            {
                "status": "submitted",
                "kernel_ref": kernel_ref,
                "source_dataset_ref": source_dataset_ref,
                "processed_source_ref": (
                    processed_kernel_ref or processed_dataset_ref
                ),
                "resume_kernel_ref": args.resume_kernel_ref,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": args.learning_rate,
                "ema_decay": args.ema_decay,
                "frames_per_chunk": args.frames_per_chunk,
                "checkpoint_every_steps": args.checkpoint_every_steps,
                "dim": args.dim,
                "depth": args.depth,
                "heads": args.heads,
                "open_vocabulary_conditioning": (
                    args.open_vocabulary_conditioning
                ),
                "lexical_holdout_fraction": (
                    args.lexical_holdout_fraction
                ),
                "minimum_lexical_sensitivity": (
                    args.minimum_lexical_sensitivity
                ),
                "lyric_semantic_weight": args.lyric_semantic_weight,
                "lyric_denoised_semantic_weight": (
                    args.lyric_denoised_semantic_weight
                ),
                "lyric_semantic_temperature": (
                    args.lyric_semantic_temperature
                ),
                "minimum_lyric_semantic_accuracy": (
                    args.minimum_lyric_semantic_accuracy
                ),
                "minimum_lyric_denoised_semantic_accuracy": (
                    args.minimum_lyric_denoised_semantic_accuracy
                ),
                "lyric_unit_semantic_weight": (
                    args.lyric_unit_semantic_weight
                ),
                "minimum_lyric_unit_accuracy": (
                    args.minimum_lyric_unit_accuracy
                ),
                "lyric_unit_denoised_semantic_weight": (
                    args.lyric_unit_denoised_semantic_weight
                ),
                "minimum_lyric_denoised_unit_accuracy": (
                    args.minimum_lyric_denoised_unit_accuracy
                ),
                "self_rollout_consistency_weight": (
                    args.self_rollout_consistency_weight
                ),
                "self_rollout_consistency_probability": (
                    args.self_rollout_consistency_probability
                ),
                "self_rollout_step_size": args.self_rollout_step_size,
                "self_rollout_solver_steps": args.self_rollout_solver_steps,
                "early_timestep_fraction": (
                    args.early_timestep_fraction
                ),
                "early_timestep_max": args.early_timestep_max,
                "semantic_pretrain_only": (
                    args.semantic_pretrain_only
                ),
                "minimum_epochs": args.minimum_epochs,
                "early_stopping_patience": (
                    args.early_stopping_patience
                ),
                "reset_optimizer": args.reset_optimizer,
                "reset_ema": args.reset_ema,
                "evaluation_only": args.evaluation_only,
                "evaluation_raw": args.evaluation_raw,
                "evaluation_steps": args.evaluation_steps,
                "evaluation_guidance_scale": (
                    args.evaluation_guidance_scale
                ),
                "evaluation_solver": args.evaluation_solver,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Local job state: {state_path.resolve()}")
    if args.submit_only:
        print(f"Submitted Kaggle kernel: {kernel_ref}")
        return

    # 5. Poll Kernel status
    # A full-scale run (e.g. 25 epochs, batch_size=8, no --max-records) can comfortably
    # exceed an hour -- a prior run of this exact script gave up at the old 240*15s=60min
    # cap while the kernel was still legitimately RUNNING, silently skipping the checkpoint
    # download. 900 iterations * 15s = 225 minutes gives real full-scale runs enough room.
    # Status matching uses the literal `KernelWorkerStatus.<NAME>` field (regex) rather than
    # a naive substring scan for "error"/"failed" -- a transient local network hiccup inside
    # the `kaggle` CLI subprocess can print those words to stderr without the kernel itself
    # having failed, which previously caused false-positive "job errored" detections.
    print("⏳ Monitoring training execution status on Kaggle...")
    status_re = re.compile(r"KernelWorkerStatus\.(\w+)", re.IGNORECASE)
    completed = False
    for _ in range(900):
        try:
            res = subprocess.run(cli + ["kernels", "status", kernel_ref], env=kaggle_env, capture_output=True, text=True, timeout=60)
            match = status_re.search(res.stdout)
            status_text = match.group(1).upper() if match else res.stdout.strip()
        except Exception as exc:
            print(f"   [WARNING] Status check failed, retrying: {exc}")
            time.sleep(15)
            continue
        print(f"   Current Status: {status_text}")
        if status_text == "COMPLETE":
            completed = True
            break
        if status_text in ("ERROR", "CANCELLED"):
            print("❌ Kaggle Kernel training failed.")
            break
        time.sleep(15)

    if completed:
        print("📥 Training complete! Downloading trained model checkpoint...")
        subprocess.run(cli + ["kernels", "output", kernel_ref, "-p", str(download_dir), "-o"], env=kaggle_env, check=True)
        checkpoint = download_dir / "my_trained_model.pt"
        if checkpoint.exists():
            # Copy checkpoint to outputs/
            final_checkpoint_path = project_root / "outputs" / "my_trained_model.pt"
            shutil.copy2(checkpoint, final_checkpoint_path)
            print(f"🎉 SUCCESS! Model checkpoint successfully downloaded to: {final_checkpoint_path.resolve()}")
        else:
            print("❌ Error: Checked completed but 'my_trained_model.pt' not found in kernel outputs.")

        best_checkpoint = download_dir / "my_trained_model.best.pt"
        if best_checkpoint.exists():
            final_best_path = project_root / "outputs" / "my_trained_model.best.pt"
            shutil.copy2(best_checkpoint, final_best_path)
            print(f"🎯 Validation-gated best checkpoint downloaded to: {final_best_path.resolve()}")
        else:
            print("ℹ️  No best_checkpoint saved -- the text-conditioning-sensitivity gate never passed this run.")

        report = download_dir / "training_report.json"
        if report.exists():
            final_report_path = job_dir / "training_report.json"
            shutil.copy2(report, final_report_path)
            data = json.loads(report.read_text(encoding="utf-8"))
            print("📊 Kết quả (kiểm soát item 8):")
            for key in (
                "record_count", "validation_record_count", "completed_epochs", "requested_epochs",
                "stopped_early", "best_epoch", "best_validation_loss", "final_validation_loss",
                "final_text_conditioning_sensitivity", "minimum_text_sensitivity", "final_loss",
                "final_lexical_holdout_sensitivity",
                "lexical_holdout_word_count",
                "open_vocabulary_conditioning",
                "elapsed_seconds",
            ):
                print(f"   {key}: {data.get(key)}")
            print(f"   Full report: {final_report_path.resolve()}")
        else:
            print("❌ Error: training_report.json not found in kernel outputs.")
    else:
        print("❌ Error: Kernel run did not complete successfully.")

if __name__ == "__main__":
    main()
