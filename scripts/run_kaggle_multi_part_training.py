"""Merge completed Kaggle preprocess outputs and train on every record.

The merge uses symlinks inside the Kaggle worker, so six processed datasets can
be treated as one without copying several gigabytes of mel tensors.  This file
is deliberately separate from the documented full-pipeline launcher.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.run_kaggle_all_parts import _old_kaggle_cli, _run_cli, _wait_for_dataset
from src.integrations.kaggle_auto import (
    kaggle_access_token,
    kaggle_auth_available,
    kaggle_auth_environment,
    load_kaggle_api_tokens,
    resolve_kaggle_username,
    write_source_zip,
)


def _parse_kernel_refs(values: list[str]) -> dict[int, str]:
    refs: dict[int, str] = {}
    for value in values:
        part_text, separator, kernel_ref = value.partition("=")
        if not separator or not part_text.isdigit() or "/" not in kernel_ref:
            raise ValueError("--kernel must use PART=OWNER/KERNEL-SLUG")
        part = int(part_text)
        if part in refs:
            raise ValueError(f"Duplicate kernel ref for part {part}")
        refs[part] = kernel_ref.strip()
    if not refs:
        raise ValueError("At least one --kernel is required")
    expected_parts = list(range(1, max(refs) + 1))
    if sorted(refs) != expected_parts:
        raise ValueError(f"Kernel parts must be consecutive from 1: expected {expected_parts}")
    return refs


def _kernel_script_content(*, source_count: int, expected_records: int, epochs: int, batch_size: int, frames_per_chunk: int, dim: int, depth: int, heads: int) -> str:
    return f'''import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def run_logged(command, label, cwd=None):
    """Persist child-process output because Kaggle can return an empty UI log."""
    result = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    output = (result.stdout or "") + "\\n" + (result.stderr or "")
    Path("/kaggle/working/" + label + ".log").write_text(output, encoding="utf-8")
    print(output, flush=True)
    if result.returncode != 0:
        raise RuntimeError(label + " failed with exit code " + str(result.returncode) + "\\n" + output[-8000:])
    return result

try:
    input_dir = Path("/kaggle/input")
    source_dataset = next(
        (path for path in input_dir.rglob("*") if path.is_dir() and "genmusic-source-" in path.name.lower()),
        None,
    )
    if source_dataset is None or not (source_dataset / "cli.py").exists():
        raise RuntimeError("Could not locate the mounted GenMusic source dataset")
    source_root = Path("/kaggle/working/GenMusic")
    shutil.copytree(source_dataset, source_root, dirs_exist_ok=True)
    os.environ["PYTHONPATH"] = str(source_root) + os.pathsep + os.environ.get("PYTHONPATH", "")

    # Verify that Kaggle really attached a GPU before doing any dataset merge.
    # A kernel can otherwise start in a CPU image even when its metadata asks
    # for a GPU, wasting time before torch reports that CUDA is unavailable.
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        message = "Kaggle did not allocate a GPU: nvidia-smi is not available"
        Path("/kaggle/working/gpu_hardware.log").write_text(message, encoding="utf-8")
        raise RuntimeError(message)
    run_logged([nvidia_smi, "-L"], "gpu_hardware")

    # Probe CUDA in a child interpreter so a CPU-only or incompatible torch can
    # be replaced before this process imports torch. This is required for P100
    # sessions when Kaggle's bundled wheel omits the GPU's sm_60 kernels.
    gpu_probe_code = (
        "import torch; "
        "print('torch=' + torch.__version__); "
        "print('cuda_available=' + repr(torch.cuda.is_available())); "
        "assert torch.cuda.is_available(), 'torch cannot access the allocated GPU'; "
        "print('gpu=' + torch.cuda.get_device_name(0)); "
        "print('capability=' + repr(torch.cuda.get_device_capability())); "
        "print('arches=' + repr(torch.cuda.get_arch_list())); "
        "print('cuda_smoke=' + repr(torch.rand(1, device='cuda').cpu().tolist()))"
    )
    initial_probe = subprocess.run(
        [sys.executable, "-c", gpu_probe_code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    Path("/kaggle/working/initial_gpu_probe.log").write_text(
        (initial_probe.stdout or "") + "\\n" + (initial_probe.stderr or ""),
        encoding="utf-8",
    )
    if initial_probe.returncode != 0:
        run_logged(
            [
                sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--no-cache-dir",
                "--force-reinstall", "torch==2.10.0", "torchaudio==2.10.0",
                "--index-url", "https://download.pytorch.org/whl/cu126",
            ],
            "install_pytorch_cuda126",
        )
    else:
        Path("/kaggle/working/install_pytorch_cuda126.log").write_text(
            "Existing torch passed a real CUDA tensor operation.", encoding="utf-8"
        )
    run_logged([sys.executable, "-c", gpu_probe_code], "gpu_preflight")

    # Import only after the child-process probe/repair so this interpreter does
    # not retain a stale CPU-only torch module after the wheel is replaced.
    import torch

    records_paths = sorted(
        path
        for path in input_dir.rglob("records.jsonl")
        if "genmusic-source-" not in str(path).lower()
    )
    if len(records_paths) != {source_count}:
        raise RuntimeError(
            "Expected {source_count} processed records.jsonl inputs, found " + str(len(records_paths))
            + ": " + repr([str(path) for path in records_paths])
        )

    combined_root = Path("/kaggle/working/combined_dataset")
    combined_mels = combined_root / "mels"
    combined_mels.mkdir(parents=True, exist_ok=True)
    combined_records = []
    source_counts = []
    required_fields = ("backing_mel_path", "vocal_mel_path", "style_embed_path")

    for source_index, records_path in enumerate(records_paths, start=1):
        source_root_dir = records_path.parent
        records = [
            json.loads(line)
            for line in records_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        source_counts.append({{"source": str(source_root_dir), "records": len(records)}})
        for record_index, record in enumerate(records, start=1):
            # Prefix both IDs and symlink names. Dataset parts currently have no
            # duplicate YouTube IDs, but this keeps future reruns collision-safe.
            record["id"] = "source%02d_%s" % (source_index, record.get("id", record_index))
            for field in required_fields:
                relative_path = record.get(field)
                source_file = source_root_dir / relative_path if relative_path else None
                if source_file is None or not source_file.is_file():
                    raise FileNotFoundError(f"Missing {{field}} for {{record['id']}}: {{source_file}}")
                destination_name = "source%02d_%s" % (source_index, source_file.name)
                destination = combined_mels / destination_name
                if not destination.exists():
                    os.symlink(source_file, destination)
                record[field] = "mels/" + destination_name
            combined_records.append(record)

    if len(combined_records) != {expected_records}:
        raise RuntimeError(
            "Expected {expected_records} combined records, found " + str(len(combined_records))
        )
    first_config = records_paths[0].parent / "config.json"
    if not first_config.is_file():
        raise FileNotFoundError("The first processed dataset has no config.json")
    shutil.copy2(first_config, combined_root / "config.json")
    (combined_root / "records.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\\n" for record in combined_records),
        encoding="utf-8",
    )
    merge_summary = {{
        "expected_sources": {source_count},
        "expected_records": {expected_records},
        "combined_records": len(combined_records),
        "sources": source_counts,
    }}
    Path("/kaggle/working/combined_summary.json").write_text(
        json.dumps(merge_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(merge_summary, ensure_ascii=False, indent=2), flush=True)

    # Generate the quality-check sample from a real training record: transcript,
    # backing frames, and MuQ-MuLan anchor all come from the same song/time span.
    # This removes the train/inference condition mismatch that previously passed
    # zero backing and no style anchor to a model trained only with real values.
    reference_record = combined_records[0]
    reference_duration = 12.0
    reference_segments = [
        segment
        for segment in (reference_record.get("segments") or [])
        if str(segment.get("text", "")).strip()
    ]
    reference_start_seconds = float(reference_segments[0].get("start", 0.0)) if reference_segments else 0.0
    lyric_lines = []
    remaining_words = 20
    for segment in reference_segments:
        if float(segment.get("start", 0.0)) >= reference_start_seconds + reference_duration:
            break
        words = str(segment.get("text", "")).strip().split()
        if not words or remaining_words <= 0:
            continue
        selected_words = words[:remaining_words]
        lyric_lines.append(" ".join(selected_words))
        remaining_words -= len(selected_words)
    if not lyric_lines:
        lyric_lines = [" ".join(str(reference_record.get("text", "")).split()[:20])]
        reference_start_seconds = 0.0
    reference_lyrics = "\\n".join(line for line in lyric_lines if line).strip()
    if not reference_lyrics:
        raise RuntimeError("The reference record has no usable transcript")

    dataset_config = json.loads((combined_root / "config.json").read_text(encoding="utf-8"))
    reference_backing_source = combined_root / reference_record["backing_mel_path"]
    full_backing = torch.load(reference_backing_source, map_location="cpu", weights_only=True).float()
    if full_backing.dim() != 2 or full_backing.shape[0] != int(dataset_config["n_mels"]):
        raise ValueError("Unexpected reference backing shape: " + repr(tuple(full_backing.shape)))
    start_frame = int(reference_start_seconds * int(dataset_config["sample_rate"]) / int(dataset_config["hop_length"]))
    duration_frames = int(reference_duration * int(dataset_config["sample_rate"]) / int(dataset_config["hop_length"]))
    reference_backing = full_backing[:, start_frame:start_frame + duration_frames]
    reference_backing_path = Path("/kaggle/working/reference_backing.pt")
    torch.save(reference_backing, reference_backing_path)
    reference_style_path = combined_root / reference_record["style_embed_path"]
    reference_style = str(reference_record.get("style") or "Vietnamese music, clear vocal")
    reference_summary = {{
        "record_id": reference_record["id"],
        "lyrics": reference_lyrics,
        "style": reference_style,
        "start_seconds": reference_start_seconds,
        "duration_seconds": reference_duration,
        "backing_mel": str(reference_backing_path),
        "style_anchor": str(reference_style_path),
    }}
    Path("/kaggle/working/reference_generation.json").write_text(
        json.dumps(reference_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Kaggle's base image may provide an older Transformers release that imports
    # successfully but has incompatible Llama block APIs. Match uv.lock exactly
    # instead of treating a shallow import probe as proof of compatibility.
    run_logged(
        [
            sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--no-cache-dir", "--upgrade",
            "transformers==5.13.1", "sentencepiece", "librosa", "vocos", "imageio-ffmpeg",
        ],
        "install_training_dependencies",
    )
    run_logged(
        [
            sys.executable,
            "-c",
            "import torch, transformers; "
            "print('torch=' + torch.__version__); "
            "print('transformers=' + transformers.__version__); "
            "print('gpu=' + torch.cuda.get_device_name(0)); "
            "print('capability=' + repr(torch.cuda.get_device_capability())); "
            "print('arches=' + repr(torch.cuda.get_arch_list())); "
            "print('cuda_smoke=' + repr(torch.rand(1, device='cuda').cpu().tolist()))",
        ],
        "training_environment",
    )

    # Exercise the exact tokenizer, MicroDiT forward pass, DataLoader, and CUDA
    # path on four real records before committing the session to 30 epochs.
    preflight_checkpoint = Path("/kaggle/working/preflight_all_parts.pt")
    run_logged(
        [
            sys.executable, str(source_root / "cli.py"), "train-self",
            "--dataset", str(combined_root),
            "--checkpoint", str(preflight_checkpoint),
            "--epochs", "1",
            "--batch-size", "4",
            "--max-records", "4",
            "--device", "cuda",
            "--frames-per-chunk", "{frames_per_chunk}",
            "--dim", "{dim}",
            "--depth", "{depth}",
            "--heads", "{heads}",
        ],
        "train_preflight",
        cwd=str(source_root),
    )
    preflight_checkpoint.unlink(missing_ok=True)

    checkpoint = Path("/kaggle/working/baseline_all_parts.pt")
    run_logged(
        [
            sys.executable, str(source_root / "cli.py"), "train-self",
            "--dataset", str(combined_root),
            "--checkpoint", str(checkpoint),
            "--epochs", "{epochs}",
            "--batch-size", "{batch_size}",
            "--device", "cuda",
            "--frames-per-chunk", "{frames_per_chunk}",
            "--dim", "{dim}",
            "--depth", "{depth}",
            "--heads", "{heads}",
        ],
        "train",
        cwd=str(source_root),
    )

    generation_dir = Path("/kaggle/working/generated_all_parts")
    run_logged(
        [
            sys.executable, str(source_root / "cli.py"), "generate-local",
            "--text", reference_lyrics,
            "--style", reference_style,
            "--duration", "12",
            "--checkpoint", str(checkpoint),
            "--steps", "64",
            "--guidance-scale", "1.5",
            "--vocoder", "vocos",
            "--device", "cuda",
            "--backing-mel", str(reference_backing_path),
            "--style-anchor", str(reference_style_path),
            "--out", str(generation_dir),
        ],
        "generate",
        cwd=str(source_root),
    )
    # The merged directory contains symlinks into /kaggle/input. It is only a
    # training view and must not be retained as a misleading standalone output.
    shutil.rmtree(combined_root)
    Path("/kaggle/working/success.txt").write_text("success", encoding="utf-8")

except Exception:
    traceback_text = traceback.format_exc()
    print(traceback_text, flush=True)
    Path("/kaggle/working/error.txt").write_text(traceback_text, encoding="utf-8")
    raise
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", action="append", default=[], metavar="PART=KERNEL_REF")
    parser.add_argument("--expected-records", type=int, default=1843)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--frames-per-chunk", type=int, default=384)
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--accelerator", default="NvidiaTeslaT4")
    parser.add_argument("--source-dataset-ref", default="")
    args = parser.parse_args()
    refs_by_part = _parse_kernel_refs(args.kernel)

    project_root = Path(__file__).resolve().parents[1]
    tokens = kaggle_auth_environment(load_kaggle_api_tokens())
    username = resolve_kaggle_username(tokens.get("KAGGLE_USERNAME"))
    if not username or not kaggle_auth_available(tokens):
        raise RuntimeError("Missing KAGGLE_USERNAME or Kaggle auth (KAGGLE_API_TOKEN=KGAT_... / legacy KAGGLE_KEY)")
    kaggle_env = {**os.environ, **tokens, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    cli = _old_kaggle_cli(tokens)

    # Refuse to submit a training job against incomplete kernel outputs. Kaggle
    # kernel sources are immutable versions, not live streams from running jobs.
    for part, ref in sorted(refs_by_part.items()):
        status_result = _run_cli(cli, ["kernels", "status", ref], kaggle_env, timeout=120)
        status = (status_result.stdout + status_result.stderr).lower()
        if status_result.returncode != 0 or "complete" not in status:
            raise RuntimeError(f"Preprocess kernel for part {part} is not complete: {ref}")

    run_id = f"multi-train-{int(time.time())}"
    run_dir = project_root / "outputs" / "kaggle_multi_part_training" / run_id
    source_dir = run_dir / "source_dataset"
    kernel_dir = run_dir / "kernel"
    source_dir.mkdir(parents=True, exist_ok=True)
    kernel_dir.mkdir(parents=True, exist_ok=True)

    source_ref = args.source_dataset_ref.strip()
    if source_ref:
        # Reuse an immutable source upload when a local submit was interrupted
        # after dataset creation but before the kernel itself was pushed.
        if "/" not in source_ref:
            raise ValueError("--source-dataset-ref must use OWNER/DATASET-SLUG")
    else:
        source_slug = f"genmusic-source-multi-train-{int(time.time())}"
        source_ref = f"{username}/{source_slug}"
        write_source_zip(project_root, source_dir / "genmusic_vn_source.zip")
        (source_dir / "dataset-metadata.json").write_text(
            json.dumps(
                # Kaggle enforces a 50-character title limit; the unique slug/id
                # still carries the full timestamp used to identify this run.
                {
                    "title": f"GenMusic multi train {int(time.time())}",
                    "id": source_ref,
                    "licenses": [{"name": "other"}],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        created = _run_cli(cli, ["datasets", "create", "-p", str(source_dir), "-r", "zip"], kaggle_env)
        if created.returncode != 0:
            raise RuntimeError("Could not create the training source dataset")
    _wait_for_dataset(cli, source_ref, kaggle_env)

    kernel_slug = f"genmusic-train-allparts-{int(time.time())}"
    kernel_ref = f"{username}/{kernel_slug}"
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
        ),
        encoding="utf-8",
    )
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
                "enable_internet": "true",
                "machine_shape": args.accelerator,
                "dataset_sources": [source_ref],
                "kernel_sources": [refs_by_part[part] for part in sorted(refs_by_part)],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    push_args = ["kernels", "push", "-p", str(kernel_dir)]
    if kaggle_access_token(tokens):
        # Kaggle CLI 1.8.4+ sends the accelerator as an explicit API override;
        # this avoids the server silently falling back to a CPU-only session.
        push_args.extend(["--accelerator", args.accelerator])
    pushed = _run_cli(cli, push_args, kaggle_env)
    push_output = (pushed.stdout + pushed.stderr).lower()
    if pushed.returncode != 0 or "kernel push error" in push_output:
        raise RuntimeError("Kaggle rejected the combined training kernel")

    state = {
        "run_id": run_id,
        "kernel_ref": kernel_ref,
        "kernel_url": f"https://www.kaggle.com/code/{kernel_ref}",
        "source_dataset_ref": source_ref,
        "processed_kernel_refs": refs_by_part,
        "expected_records": args.expected_records,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "frames_per_chunk": args.frames_per_chunk,
        "dim": args.dim,
        "depth": args.depth,
        "heads": args.heads,
        "accelerator": args.accelerator,
        "status": "submitted",
    }
    state_path = run_dir / "state.json"
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(f"Submitted: {state['kernel_url']}")
    print(f"State: {state_path}")


if __name__ == "__main__":
    main()
