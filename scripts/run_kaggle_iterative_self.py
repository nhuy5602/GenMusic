"""Run an offline, evaluated full-dataset self-training iteration on Kaggle.

This launcher is separate from the documented pipeline and from the existing
distillation launcher.  It uses only ground-truth vocal CFM because resizing a
Vocos mel directly into DiffRhythm2's private VAE latent space is not a valid
teacher target.  A completed kernel can be attached to the next invocation to
resume training after objective WER/CER evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.prepare_kaggle_offline_assets import prepare_offline_assets
from scripts.run_kaggle_all_parts import _old_kaggle_cli, _run_cli
from scripts.run_kaggle_multi_part_training import _parse_kernel_refs
from src.integrations.kaggle_auto import (
    kaggle_access_token,
    kaggle_auth_available,
    kaggle_auth_environment,
    kaggle_kernel_complete,
    load_kaggle_api_tokens,
    resolve_kaggle_username,
    write_source_zip,
)


def _kernel_script_content(
    *,
    source_count: int,
    expected_records: int,
    epochs: int,
    batch_size: int,
    frames_per_chunk: int,
    dim: int,
    depth: int,
    heads: int,
    learning_rate: float,
    style_dropout: float,
    text_dropout: float,
    text_contrastive_weight: float,
    text_contrastive_margin: float,
    text_contrastive_prob: float,
    text_sensitivity_weight: float,
    text_sensitivity_target: float,
    minimum_text_sensitivity: float,
    text_encoder: str,
    generation_target: str,
    lambda_vocal: float,
    native_ctc_weight: float,
    native_ctc_teacher_weight: float,
    native_frame_text_weight: float,
    native_frame_text_teacher_weight: float,
    native_vocal_prior_weight: float,
    vocal_structure_weight: float,
    native_prosody_weight: float,
    native_lr_multiplier: float,
    freeze_native_ctc: bool,
    dataset_validation_max_records: int,
    validation_fraction: float,
    validation_max_records: int,
    early_stopping_patience: int,
    minimum_epochs: int,
    evaluation_records: int,
    guidance_scales: str,
    resume: bool,
    online_assets: bool,
    train_only: bool,
    disable_amp: bool,
    reset_optimizer: bool,
) -> str:
    script = f'''import json
import os
import shutil
import subprocess
import sys
import tarfile
import traceback
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


def run_logged(command, label, cwd=None, env=None):
    """Stream progress live and keep the same text as a downloadable log."""
    log_path = Path("/kaggle/working") / (label + ".log")
    with log_path.open("w", encoding="utf-8") as log_stream:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        tail = []
        for line in process.stdout:
            print(line, end="", flush=True)
            log_stream.write(line)
            log_stream.flush()
            tail.append(line)
            if len(tail) > 300:
                tail.pop(0)
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(label + " failed with exit code " + str(return_code) + "\\n" + "".join(tail))


try:
    input_dir = Path("/kaggle/input")
    source_cli = next(
        (
            path
            for path in input_dir.rglob("cli.py")
            if (path.parent / "src").is_dir()
            and (path.parent / "scripts/run_kaggle_iterative_self.py").is_file()
        ),
        None,
    )
    if source_cli is None:
        raise RuntimeError("Could not locate the mounted GenMusic source dataset")
    source_root = Path("/kaggle/working/GenMusic")
    shutil.copytree(source_cli.parent, source_root, dirs_exist_ok=True)

    assets_root = Path("/kaggle/working/offline_assets")
    mounted_assets_root = next(
        (
            path.parent
            for path in input_dir.rglob("models")
            if (path / "xphonebert").is_dir()
            and (path / "charsiu_g2p").is_dir()
            and (path.parent / "whisper/small.pt").is_file()
        ),
        None,
    )
    if mounted_assets_root is not None:
        # Kaggle commonly expands an uploaded tar while ingesting the dataset,
        # so the kernel mount contains its children rather than the tar itself.
        shutil.copytree(mounted_assets_root, assets_root, dirs_exist_ok=True)
    else:
        offline_tar = next(input_dir.rglob("genmusic_offline_assets.tar"), None)
        if offline_tar is None:
            raise RuntimeError("Could not locate extracted or archived offline assets")
        assets_root.mkdir(parents=True, exist_ok=True)
        with tarfile.open(offline_tar) as archive:
            archive.extractall(assets_root)

    os.environ.update({{
        "PYTHONPATH": str(source_root) + os.pathsep + os.environ.get("PYTHONPATH", ""),
        "GENMUSIC_XPHONEBERT_PATH": str(assets_root / "models/xphonebert"),
        "GENMUSIC_CHARSIU_G2P_PATH": str(assets_root / "models/charsiu_g2p"),
        "GENMUSIC_BYT5_PATH": str(assets_root / "models/byt5"),
        "GENMUSIC_VOCOS_PATH": str(assets_root / "models/vocos"),
        "GENMUSIC_WHISPER_MODEL_PATH": str(assets_root / "whisper/small.pt"),
    }})
    shutil.copy2(assets_root / "vie-c.tsv", source_root / "vie-c.tsv")

    wheelhouse = assets_root / "wheelhouse"
    source_archives = list(wheelhouse.glob("*.tar.gz"))
    if source_archives:
        run_logged(
            [
                sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
                "--no-index", "--find-links", str(wheelhouse), "--no-deps",
                "--no-build-isolation", "vocos==0.1.0", "encodec==0.1.1",
                "text2phonemesequence==0.1.4", "segments==2.4.0", "csvw",
                "isodate", "python-dateutil", "rfc3986<2", "uritemplate", "babel",
                "language-tags", "rdflib", "termcolor", "jsonschema",
                "openai-whisper==20250625", "more-itertools", "tiktoken",
            ],
            "install_offline_dependencies",
        )
    else:
        # Kaggle expands source distributions but currently leaves wheels
        # intact. Install compatible wheels directly, then expose the three
        # expanded pure-Python source packages through PYTHONPATH.
        interpreter_tag = "cp%d%d" % (sys.version_info.major, sys.version_info.minor)
        wheel_files = [
            path
            for path in wheelhouse.glob("*.whl")
            if "-cp3" not in path.name or interpreter_tag in path.name
        ]
        if not wheel_files:
            raise RuntimeError("Kaggle mount has neither source archives nor compatible wheels")
        run_logged(
            [
                sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
                "--no-index", "--no-deps", *[str(path) for path in wheel_files],
            ],
            "install_offline_wheels",
        )
        package_names = ("encodec", "text2phonemesequence", "whisper")
        vendor_roots = []
        for package_name in package_names:
            candidates = [
                path
                for path in wheelhouse.rglob(package_name)
                if path.is_dir() and (path / "__init__.py").is_file()
            ]
            if candidates:
                parent = str(candidates[0].parent)
            else:
                raise RuntimeError("Offline package was not extracted: " + package_name)
            if parent not in vendor_roots:
                vendor_roots.append(parent)
        os.environ["PYTHONPATH"] = os.pathsep.join(
            [str(source_root), *vendor_roots, os.environ.get("PYTHONPATH", "")]
        )
        sys.path[:0] = [str(source_root), *vendor_roots]
        Path("/kaggle/working/install_offline_dependencies.log").write_text(
            "Using Kaggle-extracted package roots:\\n" + "\\n".join(vendor_roots),
            encoding="utf-8",
        )

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        raise RuntimeError("Kaggle did not allocate a GPU")
    run_logged([nvidia_smi, "-L"], "gpu_hardware")
    gpu_probe = (
        "import torch,transformers; "
        "print('torch=' + torch.__version__); print('transformers=' + transformers.__version__); "
        "assert torch.cuda.is_available(); print('gpu=' + torch.cuda.get_device_name(0)); "
        "print('capability=' + repr(torch.cuda.get_device_capability())); "
        "print('arches=' + repr(torch.cuda.get_arch_list())); "
        "print('cuda_smoke=' + repr(torch.rand(1, device='cuda').cpu().tolist()))"
    )
    run_logged([sys.executable, "-c", gpu_probe], "gpu_preflight")

    records_paths = sorted(
        path
        for path in input_dir.rglob("records.jsonl")
        if "genmusic-source-" not in str(path).lower()
    )
    if len(records_paths) != {source_count}:
        raise RuntimeError(
            "Expected {source_count} processed inputs, found " + str(len(records_paths))
            + ": " + repr([str(path) for path in records_paths])
        )

    combined_root = Path("/kaggle/working/combined_dataset")
    combined_mels = combined_root / "mels"
    combined_mels.mkdir(parents=True, exist_ok=True)
    combined_records = []
    source_counts = []
    required_fields = ("backing_mel_path", "vocal_mel_path", "style_embed_path")
    for source_index, records_path in enumerate(records_paths, start=1):
        source_dir = records_path.parent
        records = [
            json.loads(line)
            for line in records_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        source_counts.append({{"source": str(source_dir), "records": len(records)}})
        for record_index, record in enumerate(records, start=1):
            record["id"] = "source%02d_%s" % (source_index, record.get("id", record_index))
            for field in required_fields:
                relative_path = record.get(field)
                source_file = source_dir / relative_path if relative_path else None
                if source_file is None or not source_file.is_file():
                    raise FileNotFoundError("Missing %s for %s: %s" % (field, record["id"], source_file))
                destination = combined_mels / ("source%02d_%s" % (source_index, source_file.name))
                if not destination.exists():
                    os.symlink(source_file, destination)
                record[field] = "mels/" + destination.name
            combined_records.append(record)

    if len(combined_records) != {expected_records}:
        raise RuntimeError("Expected {expected_records} combined records, found " + str(len(combined_records)))
    shutil.copy2(records_paths[0].parent / "config.json", combined_root / "config.json")
    (combined_root / "records.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\\n" for record in combined_records),
        encoding="utf-8",
    )
    Path("/kaggle/working/combined_summary.json").write_text(
        json.dumps({{
            "expected_records": {expected_records},
            "combined_records": len(combined_records),
            "sources": source_counts,
        }}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Keep the one-epoch smoke report out of /kaggle/working so it can never be
    # mistaken for the full run's training_report.json during a resume.
    preflight_checkpoint = Path("/kaggle/working/preflight/preflight_self.pt")
    preflight_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    preflight_command = [
        sys.executable, str(source_root / "cli.py"), "train-self",
        "--dataset", str(combined_root), "--checkpoint", str(preflight_checkpoint),
        "--epochs", "1", "--batch-size", "2", "--max-records", "4",
        "--frames-per-chunk", "{frames_per_chunk}", "--dim", "{dim}",
        "--depth", "{depth}", "--heads", "{heads}", "--device", "cuda",
        "--validation-fraction", "0", "--style-dropout", "{style_dropout}",
        "--text-dropout", "{text_dropout}",
        "--text-encoder", "{text_encoder}",
        "--generation-target", "{generation_target}",
    ]
    if {str(disable_amp)}:
        preflight_command.append("--disable-amp")
    run_logged(
        preflight_command,
        "train_preflight",
        cwd=str(source_root),
        env=os.environ,
    )
    shutil.rmtree(preflight_checkpoint.parent, ignore_errors=True)

    checkpoint = Path("/kaggle/working/self_all_parts.pt")
    resume_enabled = {str(resume)}
    if resume_enabled:
        previous_checkpoints = sorted(
            path for path in input_dir.rglob("self_all_parts.pt") if path.is_file()
        )
        if not previous_checkpoints:
            raise RuntimeError("Resume kernel was attached but self_all_parts.pt was not found")
        shutil.copy2(previous_checkpoints[-1], checkpoint)

    train_command = [
        sys.executable, str(source_root / "cli.py"), "train-self",
        "--dataset", str(combined_root), "--checkpoint", str(checkpoint),
        "--epochs", "{epochs}", "--batch-size", "{batch_size}",
        "--learning-rate", "{learning_rate}", "--frames-per-chunk", "{frames_per_chunk}",
        "--dim", "{dim}", "--depth", "{depth}", "--heads", "{heads}",
        "--style-dropout", "{style_dropout}", "--text-dropout", "{text_dropout}",
        "--text-contrastive-weight", "{text_contrastive_weight}",
        "--text-contrastive-margin", "{text_contrastive_margin}",
        "--text-contrastive-prob", "{text_contrastive_prob}",
        "--text-sensitivity-weight", "{text_sensitivity_weight}",
        "--text-sensitivity-target", "{text_sensitivity_target}",
        "--minimum-text-sensitivity", "{minimum_text_sensitivity}",
        "--text-encoder", "{text_encoder}",
        "--generation-target", "{generation_target}",
        "--lambda-vocal", "{lambda_vocal}",
        "--native-ctc-weight", "{native_ctc_weight}",
        "--native-ctc-teacher-weight", "{native_ctc_teacher_weight}",
        "--native-frame-text-weight", "{native_frame_text_weight}",
        "--native-frame-text-teacher-weight", "{native_frame_text_teacher_weight}",
        "--native-vocal-prior-weight", "{native_vocal_prior_weight}",
        "--vocal-structure-weight", "{vocal_structure_weight}",
        "--native-prosody-weight", "{native_prosody_weight}",
        "--native-lr-multiplier", "{native_lr_multiplier}",
        "--dataset-validation-max-records", "{dataset_validation_max_records}",
        "--validation-fraction", "{validation_fraction}",
        "--validation-max-records", "{validation_max_records}",
        "--early-stopping-patience", "{early_stopping_patience}",
        "--minimum-epochs", "{minimum_epochs}", "--device", "cuda",
        "--save-every-epoch", "--checkpoint-every-steps", "200", "--log-every-steps", "10",
    ]
    if resume_enabled:
        train_command.append("--resume")
    if {str(reset_optimizer)}:
        train_command.append("--reset-optimizer")
    if {str(disable_amp)}:
        train_command.append("--disable-amp")
    if {str(freeze_native_ctc)}:
        train_command.append("--freeze-native-ctc")
    run_logged(train_command, "train_full", cwd=str(source_root), env=os.environ)

    if {str(train_only)}:
        training_report_path = Path("/kaggle/working/training_report.json")
        if not training_report_path.is_file():
            mounted_reports = list(input_dir.rglob("training_report.json"))
            if mounted_reports:
                def completed_epochs(path):
                    try:
                        return int(json.loads(path.read_text(encoding="utf-8")).get("completed_epochs", 0))
                    except Exception:
                        return 0
                shutil.copy2(max(mounted_reports, key=completed_epochs), training_report_path)
        report = (
            json.loads(training_report_path.read_text(encoding="utf-8"))
            if training_report_path.is_file()
            else {{"status": "complete", "completed_epochs": {epochs}}}
        )
        Path("/kaggle/working/train_phase_result.json").write_text(
            json.dumps({{
                "phase": "train",
                "checkpoint": str(checkpoint),
                "completed_epochs": report.get("completed_epochs", {epochs}),
                "best_epoch": report.get("best_epoch"),
                "best_validation_loss": report.get("best_validation_loss"),
                "final_text_conditioning_sensitivity": report.get("final_text_conditioning_sensitivity"),
                "final_native_ctc_validation": report.get("final_native_ctc_validation"),
                "native_generation": report.get("native_generation"),
            }}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        shutil.rmtree(combined_root, ignore_errors=True)
        shutil.rmtree(source_root, ignore_errors=True)
        Path("/kaggle/working/success.txt").write_text("success", encoding="utf-8")
        sys.exit(0)

    best_checkpoint = checkpoint.with_name(checkpoint.stem + ".best" + checkpoint.suffix)
    evaluation_checkpoint = best_checkpoint if best_checkpoint.is_file() else checkpoint
    evaluation_dir = Path("/kaggle/working/quality_evaluation")
    run_logged(
        [
            sys.executable, str(source_root / "scripts/evaluate_generation_quality.py"),
            str(evaluation_checkpoint), str(combined_root), str(evaluation_dir),
            "{evaluation_records}", "--whisper-model", "small", "--duration", "8", "--steps", "64",
            "--guidance-scales", "{guidance_scales}",
        ],
        "quality_evaluation",
        cwd=str(source_root),
        env=os.environ,
    )
    quality_report = json.loads((evaluation_dir / "quality_report.json").read_text(encoding="utf-8"))
    summary = quality_report.get("summary") or {{}}
    samples = quality_report.get("samples") or []
    ranked_samples = sorted(
        samples,
        key=lambda item: float((item.get("generated_asr") or {{}}).get("word_accuracy", 0.0)),
        reverse=True,
    )
    if ranked_samples:
        best_id = ranked_samples[0]["id"]
        best_wav = evaluation_dir / (best_id + "_generated.wav")
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg and best_wav.is_file():
            run_logged(
                [ffmpeg, "-y", "-i", str(best_wav), "-codec:a", "libmp3lame", "-q:a", "2", "/kaggle/working/best_generated.mp3"],
                "encode_best_mp3",
            )
    training_report = json.loads(Path("/kaggle/working/training_report.json").read_text(encoding="utf-8"))
    report_plots_dir = Path("/kaggle/working/report_plots")
    run_logged(
        [
            sys.executable, str(source_root / "scripts/create_kaggle_report_plots.py"),
            "/kaggle/working/training_report.json",
            str(evaluation_dir / "quality_report.json"),
            str(report_plots_dir),
        ],
        "create_report_plots",
        cwd=str(source_root),
        env=os.environ,
    )
    report_artifacts = sorted(
        path.relative_to(Path("/kaggle/working")).as_posix()
        for path in report_plots_dir.iterdir()
        if path.is_file()
    )
    Path("/kaggle/working/iteration_result.json").write_text(
        json.dumps({{
            "checkpoint": str(evaluation_checkpoint),
            "intelligibility_pass": bool(summary.get("intelligibility_pass")),
            "quality_summary": summary,
            "training_summary": training_report,
            "report_artifacts": report_artifacts,
            "recommended_next_target_epochs": int(training_report.get("completed_epochs", {epochs})) + 8,
        }}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    shutil.rmtree(combined_root)
    Path("/kaggle/working/success.txt").write_text("success", encoding="utf-8")
except Exception:
    traceback_text = traceback.format_exc()
    print(traceback_text, flush=True)
    Path("/kaggle/working/error.txt").write_text(traceback_text, encoding="utf-8")
    raise
'''
    if text_encoder == "native_utf8" and train_only:
        # Native train-only kernels need neither XPhoneBERT/Charsiu nor a TTS,
        # vocoder, or Whisper. Starting the GPU only after dependency setup
        # avoids spending quota on downloads that cannot affect training.
        offline_start = script.index('    assets_root = Path("/kaggle/working/offline_assets")')
        offline_end = script.index('    nvidia_smi = shutil.which("nvidia-smi")')
        native_setup = '''    Path("/kaggle/working/native_text_frontend.txt").write_text(
        "native_utf8: no pretrained text encoder or TTS loaded",
        encoding="utf-8",
    )

'''
        return script[:offline_start] + native_setup + script[offline_end:]
    if not online_assets:
        return script

    # Keep an Internet-backed fallback separate from the reproducible offline
    # path. This avoids uploading the 2.1 GB bundle over a slow local uplink;
    # Kaggle downloads the same public dependencies inside its own runtime.
    script = script.replace(
        'os.environ["HF_HUB_OFFLINE"] = "1"\n'
        'os.environ["TRANSFORMERS_OFFLINE"] = "1"\n',
        "",
    )
    offline_start = script.index('    assets_root = Path("/kaggle/working/offline_assets")')
    offline_end = script.index('    nvidia_smi = shutil.which("nvidia-smi")')
    online_setup = '''    run_logged(
        [
            sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
            "vocos==0.1.0", "encodec==0.1.1",
            "text2phonemesequence==0.1.4", "segments==2.4.0",
            "openai-whisper==20250625",
        ],
        "install_online_dependencies",
    )
    import urllib.request
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/lingjzhu/CharsiuG2P/main/dicts/vie-c.tsv",
        source_root / "vie-c.tsv",
    )
    Path("/kaggle/working/online_assets_ready.txt").write_text(
        "Online package install and Vietnamese G2P dictionary download succeeded.",
        encoding="utf-8",
    )

'''
    return script[:offline_start] + online_setup + script[offline_end:]


def _create_dataset(
    *,
    cli: list[str],
    env: dict[str, str],
    upload_dir: Path,
    dataset_ref: str,
    title: str,
    expected_marker: str,
) -> None:
    (upload_dir / "dataset-metadata.json").write_text(
        json.dumps(
            {"title": title[:50], "id": dataset_ref, "licenses": [{"name": "other"}]},
            indent=2,
        ),
        encoding="utf-8",
    )
    # The reusable offline bundle is ~2.1 GB. Slower uplinks can legitimately
    # exceed the generic 15-minute CLI timeout even though bytes are still
    # flowing, so give dataset creation its own one-hour transfer budget.
    result = _run_cli(
        cli,
        ["datasets", "create", "-p", str(upload_dir), "-r", "zip"],
        env,
        timeout=3_600,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Could not create Kaggle dataset {dataset_ref}")
    _wait_for_dataset_visible(cli, dataset_ref, env, expected_marker=expected_marker)


def _wait_for_dataset_visible(
    cli: list[str],
    dataset_ref: str,
    env: dict[str, str],
    *,
    expected_marker: str,
) -> None:
    """Wait for either status=ready or a fully indexed expected file.

    The modern KGAT endpoint can temporarily return 403 from ``datasets
    status`` for a newly-created private dataset even while ``datasets files``
    already lists its content. Accepting the expected immutable file avoids a
    false 15-minute timeout while still proving the upload is indexed.
    """
    for _ in range(180):
        status_result = _run_cli(cli, ["datasets", "status", dataset_ref], env, timeout=120)
        status = (status_result.stdout + status_result.stderr).lower()
        if status_result.returncode == 0 and "ready" in status:
            time.sleep(15)
            return
        files_result = _run_cli(
            cli,
            ["datasets", "files", dataset_ref, "--page-size", "200"],
            env,
            timeout=120,
        )
        files_text = files_result.stdout + files_result.stderr
        if files_result.returncode == 0 and expected_marker in files_text:
            time.sleep(15)
            return
        time.sleep(5)
    raise TimeoutError(f"Kaggle dataset did not become visible: {dataset_ref}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", action="append", default=[], metavar="PART=KERNEL_REF")
    parser.add_argument("--expected-records", type=int, default=1843)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--frames-per-chunk", type=int, default=384)
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--style-dropout", type=float, default=0.5)
    parser.add_argument("--text-dropout", type=float, default=0.1)
    parser.add_argument("--text-contrastive-weight", type=float, default=0.08)
    parser.add_argument("--text-contrastive-margin", type=float, default=0.03)
    parser.add_argument("--text-contrastive-prob", type=float, default=0.5)
    parser.add_argument("--text-sensitivity-weight", type=float, default=2.0)
    parser.add_argument("--text-sensitivity-target", type=float, default=0.20)
    parser.add_argument("--minimum-text-sensitivity", type=float, default=0.18)
    parser.add_argument(
        "--text-encoder",
        default="native_utf8",
        choices=("native_utf8", "pretrained_xphonebert"),
    )
    parser.add_argument(
        "--generation-target",
        default="joint_stems",
        choices=("joint_stems", "full_mix"),
    )
    parser.add_argument(
        "--lambda-vocal",
        type=float,
        default=1.0,
        help="Direct and auxiliary loss weight for the vocal half of joint stems.",
    )
    parser.add_argument("--native-ctc-weight", type=float, default=0.0)
    parser.add_argument("--native-ctc-teacher-weight", type=float, default=0.0)
    parser.add_argument("--native-frame-text-weight", type=float, default=0.0)
    parser.add_argument("--native-frame-text-teacher-weight", type=float, default=0.0)
    parser.add_argument("--native-vocal-prior-weight", type=float, default=0.0)
    parser.add_argument("--vocal-structure-weight", type=float, default=0.0)
    parser.add_argument("--native-prosody-weight", type=float, default=0.0)
    parser.add_argument("--native-lr-multiplier", type=float, default=10.0)
    parser.add_argument(
        "--freeze-native-ctc",
        action="store_true",
        help="Freeze a separately pretrained native recognizer during diffusion continuation.",
    )
    parser.add_argument(
        "--disable-amp",
        action="store_true",
        help="Use FP32 CUDA training when FP16 backward remains unstable.",
    )
    parser.add_argument(
        "--reset-optimizer",
        action="store_true",
        help="Resume model/EMA but restart AdamW and LR schedule for a new objective phase.",
    )
    parser.add_argument("--dataset-validation-max-records", type=int, default=128)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--validation-max-records", type=int, default=96)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--minimum-epochs", type=int, default=10)
    parser.add_argument("--evaluation-records", type=int, default=6)
    parser.add_argument("--guidance-scales", default="1.0,2.0,3.0,4.0")
    parser.add_argument(
        "--train-only",
        action="store_true",
        help=(
            "Stop after checkpoint/training_report output. Run quality and plot phases "
            "with their dedicated scripts so they can be retried independently."
        ),
    )
    parser.add_argument("--accelerator", default="NvidiaTeslaP100")
    parser.add_argument(
        "--session-timeout-seconds",
        type=int,
        default=14_400,
        help="Cap Kaggle runtime while leaving time for validation, Whisper evaluation, and plots.",
    )
    parser.add_argument("--source-dataset-ref", default="")
    parser.add_argument("--offline-assets-ref", default="")
    parser.add_argument(
        "--online-assets",
        action="store_true",
        help="Download public model assets inside Kaggle instead of uploading the offline bundle.",
    )
    parser.add_argument("--resume-kernel-ref", default="")
    parser.add_argument(
        "--resume-dataset-ref",
        default="",
        help=(
            "Private Kaggle dataset containing self_all_parts.pt. Use this when a timed-out "
            "kernel's checkpoint was downloaded and re-uploaded as a dataset."
        ),
    )
    parser.add_argument(
        "--kernel-slug",
        default="",
        help="Reuse an existing kernel slug to replace a wasteful running version.",
    )
    args = parser.parse_args()
    if args.session_timeout_seconds < 600:
        raise ValueError("--session-timeout-seconds must be at least 600")
    if args.resume_kernel_ref and args.resume_dataset_ref:
        raise ValueError("Choose either --resume-kernel-ref or --resume-dataset-ref, not both")
    refs_by_part = _parse_kernel_refs(args.kernel)

    project_root = Path(__file__).resolve().parents[1]
    tokens = kaggle_auth_environment(load_kaggle_api_tokens())
    username = resolve_kaggle_username(tokens.get("KAGGLE_USERNAME"))
    if not username or not kaggle_auth_available(tokens):
        raise RuntimeError("Missing Kaggle username/access token")
    kaggle_env = {**os.environ, **tokens, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    cli = _old_kaggle_cli(tokens)

    refs_to_check = list(refs_by_part.values())
    if args.resume_kernel_ref:
        refs_to_check.append(args.resume_kernel_ref)
    for ref in refs_to_check:
        if not kaggle_kernel_complete(ref):
            raise RuntimeError(f"Required kernel is not complete: {ref}")

    resume_dataset_ref = args.resume_dataset_ref.strip()
    if resume_dataset_ref:
        _wait_for_dataset_visible(
            cli,
            resume_dataset_ref,
            kaggle_env,
            expected_marker="self_all_parts.pt",
        )

    timestamp = int(time.time())
    run_id = f"iterative-self-{timestamp}"
    run_dir = project_root / "outputs/kaggle_iterative_self" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    assets_ref = args.offline_assets_ref.strip()
    if args.online_assets and assets_ref:
        raise ValueError("Choose either --online-assets or --offline-assets-ref, not both")
    needs_asset_dataset = not (
        args.text_encoder == "native_utf8" and args.train_only
    )
    if not args.online_assets and needs_asset_dataset:
        if not assets_ref:
            assets_ref = f"{username}/genmusic-offline-assets-{timestamp}"
            bundle = prepare_offline_assets(project_root / "outputs/kaggle_offline_assets/current")
            assets_upload = run_dir / "assets_dataset"
            assets_upload.mkdir(parents=True, exist_ok=True)
            os.link(bundle, assets_upload / bundle.name)
            _create_dataset(
                cli=cli,
                env=kaggle_env,
                upload_dir=assets_upload,
                dataset_ref=assets_ref,
                title=f"GenMusic offline assets {timestamp}",
                expected_marker="whisper/small.pt",
            )
        else:
            _wait_for_dataset_visible(
                cli,
                assets_ref,
                kaggle_env,
                expected_marker="whisper/small.pt",
            )

    source_ref = args.source_dataset_ref.strip()
    if not source_ref:
        source_ref = f"{username}/genmusic-source-iter-self-{timestamp}"
        source_upload = run_dir / "source_dataset"
        source_upload.mkdir(parents=True, exist_ok=True)
        write_source_zip(project_root, source_upload / "genmusic_vn_source.zip")
        _create_dataset(
            cli=cli,
            env=kaggle_env,
            upload_dir=source_upload,
            dataset_ref=source_ref,
            title=f"GenMusic iterative self {timestamp}",
            expected_marker="cli.py",
        )
    else:
        _wait_for_dataset_visible(
            cli,
            source_ref,
            kaggle_env,
            expected_marker="cli.py",
        )

    kernel_slug = args.kernel_slug.strip() or f"genmusic-iter-self-{timestamp}"
    kernel_ref = f"{username}/{kernel_slug}"
    kernel_dir = run_dir / "kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / "run_training.py").write_text(
        _kernel_script_content(
            source_count=len(refs_by_part),
            expected_records=args.expected_records,
            epochs=args.epochs,
            batch_size=args.batch_size,
            frames_per_chunk=args.frames_per_chunk,
            dim=args.dim,
            depth=args.depth,
            heads=args.heads,
            learning_rate=args.learning_rate,
            style_dropout=args.style_dropout,
            text_dropout=args.text_dropout,
            text_contrastive_weight=args.text_contrastive_weight,
            text_contrastive_margin=args.text_contrastive_margin,
            text_contrastive_prob=args.text_contrastive_prob,
            text_sensitivity_weight=args.text_sensitivity_weight,
            text_sensitivity_target=args.text_sensitivity_target,
            minimum_text_sensitivity=args.minimum_text_sensitivity,
            text_encoder=args.text_encoder,
            generation_target=args.generation_target,
            lambda_vocal=args.lambda_vocal,
            native_ctc_weight=args.native_ctc_weight,
            native_ctc_teacher_weight=args.native_ctc_teacher_weight,
            native_frame_text_weight=args.native_frame_text_weight,
            native_frame_text_teacher_weight=(
                args.native_frame_text_teacher_weight
            ),
            native_vocal_prior_weight=args.native_vocal_prior_weight,
            vocal_structure_weight=args.vocal_structure_weight,
            native_prosody_weight=args.native_prosody_weight,
            native_lr_multiplier=args.native_lr_multiplier,
            freeze_native_ctc=args.freeze_native_ctc,
            dataset_validation_max_records=args.dataset_validation_max_records,
            validation_fraction=args.validation_fraction,
            validation_max_records=args.validation_max_records,
            early_stopping_patience=args.early_stopping_patience,
            minimum_epochs=args.minimum_epochs,
            evaluation_records=args.evaluation_records,
            guidance_scales=args.guidance_scales,
            resume=bool(args.resume_kernel_ref or resume_dataset_ref),
            online_assets=args.online_assets,
            train_only=args.train_only,
            disable_amp=args.disable_amp,
            reset_optimizer=args.reset_optimizer,
        ),
        encoding="utf-8",
    )
    kernel_sources = [refs_by_part[part] for part in sorted(refs_by_part)]
    if args.resume_kernel_ref:
        kernel_sources.append(args.resume_kernel_ref)
    dataset_sources = [source_ref] + ([assets_ref] if assets_ref else [])
    if resume_dataset_ref:
        dataset_sources.append(resume_dataset_ref)
    (kernel_dir / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": kernel_ref,
                "title": kernel_slug,
                "code_file": "run_training.py",
                "language": "python",
                "kernel_type": "script",
                "is_private": "true",
                "enable_gpu": "true",
                "enable_internet": "true" if args.online_assets else "false",
                "machine_shape": args.accelerator,
                "dataset_sources": dataset_sources,
                "kernel_sources": kernel_sources,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    state = {
        "run_id": run_id,
        "kernel_ref": kernel_ref,
        "kernel_url": f"https://www.kaggle.com/code/{kernel_ref}",
        "source_dataset_ref": source_ref,
        "offline_assets_ref": assets_ref or None,
        "online_assets": args.online_assets,
        "processed_kernel_refs": refs_by_part,
        "resume_kernel_ref": args.resume_kernel_ref or None,
        "resume_dataset_ref": resume_dataset_ref or None,
        "expected_records": args.expected_records,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "frames_per_chunk": args.frames_per_chunk,
        "dim": args.dim,
        "depth": args.depth,
        "heads": args.heads,
        "learning_rate": args.learning_rate,
        "style_dropout": args.style_dropout,
        "text_dropout": args.text_dropout,
        "text_contrastive_weight": args.text_contrastive_weight,
        "text_contrastive_margin": args.text_contrastive_margin,
        "text_contrastive_prob": args.text_contrastive_prob,
        "text_sensitivity_weight": args.text_sensitivity_weight,
        "text_sensitivity_target": args.text_sensitivity_target,
        "minimum_text_sensitivity": args.minimum_text_sensitivity,
        "text_encoder": args.text_encoder,
        "lambda_vocal": args.lambda_vocal,
        "native_ctc_weight": args.native_ctc_weight,
        "generation_target": args.generation_target,
        "native_ctc_teacher_weight": args.native_ctc_teacher_weight,
        "native_frame_text_weight": args.native_frame_text_weight,
        "native_frame_text_teacher_weight": (
            args.native_frame_text_teacher_weight
        ),
        "native_vocal_prior_weight": args.native_vocal_prior_weight,
        "vocal_structure_weight": args.vocal_structure_weight,
        "native_prosody_weight": args.native_prosody_weight,
        "native_lr_multiplier": args.native_lr_multiplier,
        "freeze_native_ctc": args.freeze_native_ctc,
        "dataset_validation_max_records": args.dataset_validation_max_records,
        "validation_fraction": args.validation_fraction,
        "validation_max_records": args.validation_max_records,
        "early_stopping_patience": args.early_stopping_patience,
        "minimum_epochs": args.minimum_epochs,
        "evaluation_records": args.evaluation_records,
        "guidance_scales": args.guidance_scales,
        "train_only": args.train_only,
        "disable_amp": args.disable_amp,
        "reset_optimizer": args.reset_optimizer,
        "accelerator": args.accelerator,
        "session_timeout_seconds": args.session_timeout_seconds,
        "status": "prepared",
    }
    state_path = run_dir / "state.json"
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    # Keep a bounded reservation, but include enough headroom for Whisper and
    # report generation after the last training checkpoint is written.
    push_args = [
        "kernels",
        "push",
        "-p",
        str(kernel_dir),
        "--timeout",
        str(args.session_timeout_seconds),
    ]
    if kaggle_access_token(tokens):
        push_args.extend(["--accelerator", args.accelerator])
    pushed = _run_cli(cli, push_args, kaggle_env)
    push_text = pushed.stdout + pushed.stderr
    if pushed.returncode != 0 or "kernel push error" in push_text.lower():
        state.update(
            {
                "status": "submit_failed",
                "submit_returncode": pushed.returncode,
                "submit_output_tail": push_text[-4000:],
            }
        )
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        raise RuntimeError(f"Kaggle rejected the iterative self-training kernel; state: {state_path}")

    state["status"] = "submitted"
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(f"Submitted: {state['kernel_url']}")
    print(f"State: {state_path}")


if __name__ == "__main__":
    main()
