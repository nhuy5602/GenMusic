from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

from .kaggle_auto import (
    KaggleJobConfig,
    _commands,
    _history_item,
    _now,
    _run,
    _summarize_cli_error,
    _wait_for_dataset_ready,
    _write_source_zip,
    _write_state,
    kaggle_cli_command,
    kaggle_readiness,
    make_run_id,
    resolve_kaggle_username,
    slugify,
)
from .trained_text_model import DEFAULT_LOCAL_MODEL_PATH
from .training_dataset import generate_training_records, load_training_records, write_training_jsonl


QUOTA_MARKERS = ("quota", "usage", "limit", "exceeded", "gpu", "not enough", "capacity")


def submit_text_model_training_job(
    *,
    output_root: str | Path = "outputs/model_training",
    sample_count: int = 480,
    seed: int = 42,
    extra_datasets: list[str | Path] | None = None,
    config: KaggleJobConfig | None = None,
    local_model_path: str | Path = DEFAULT_LOCAL_MODEL_PATH,
) -> dict[str, Any]:
    config = config or KaggleJobConfig(machine_shape="CPU")
    state = stage_text_model_training_job(
        output_root=output_root,
        sample_count=sample_count,
        seed=seed,
        extra_datasets=extra_datasets or [],
        config=config,
        local_model_path=local_model_path,
    )
    if not config.submit:
        state["status"] = "staged"
        state["messages"].append("Text model training job staged locally. Submit is disabled.")
        _write_state(state)
        return state

    readiness = kaggle_readiness(config.username)
    state["kaggle_ready"] = readiness["ready"]
    state["messages"].extend(readiness["messages"])
    _write_state(state)
    if not readiness["ready"]:
        state["status"] = "needs_setup"
        state["messages"].append("Install/configure Kaggle API, then submit the generated training commands.")
        _write_state(state)
        return state

    return submit_training_kaggle_job(
        state,
        wait=config.wait,
        poll_seconds=config.poll_seconds,
        timeout_seconds=config.timeout_seconds,
    )


def stage_text_model_training_job(
    *,
    output_root: str | Path,
    sample_count: int,
    seed: int,
    extra_datasets: list[str | Path],
    config: KaggleJobConfig,
    local_model_path: str | Path,
) -> dict[str, Any]:
    run_id = make_run_id(f"text-model-training-{seed}-{sample_count}")
    username = resolve_kaggle_username(config.username) or "YOUR_KAGGLE_USERNAME"
    run_dir = Path(output_root) / run_id
    job_dir = run_dir / "kaggle_train_model"
    dataset_dir = job_dir / "dataset"
    kernel_dir = job_dir / "kernel"
    download_dir = job_dir / "downloaded_output"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    kernel_dir.mkdir(parents=True, exist_ok=True)
    download_dir.mkdir(parents=True, exist_ok=True)

    generated = generate_training_records(sample_count, seed=seed)
    extra_records = load_training_records(extra_datasets)
    training_records = generated + extra_records
    training_jsonl = dataset_dir / "training_data.jsonl"
    write_training_jsonl(training_records, training_jsonl)
    _write_source_zip(dataset_dir / "genmusic_vn_source.zip")

    dataset_slug = slugify(f"genmusic-vn-train-data-{run_id}", max_length=48)
    kernel_slug = slugify(f"genmusic-vn-train-model-{run_id}", max_length=48)
    dataset_ref = f"{username}/{dataset_slug}"
    kernel_ref = f"{username}/{kernel_slug}"
    request = {
        "run_id": run_id,
        "job_kind": "text_model_training",
        "sample_count": sample_count,
        "extra_record_count": len(extra_records),
        "total_record_count": len(training_records),
        "seed": seed,
        "local_model_path": str(local_model_path),
        "created_at": _now(),
    }
    (run_dir / "training_request.json").write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")
    (dataset_dir / "training_request.json").write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")
    (dataset_dir / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": dataset_slug,
                "id": dataset_ref,
                "licenses": [{"name": "other"}],
                "subtitle": "Vietnamese text-to-music supervised training data for GenMusic VN.",
                "description": "Synthetic and optional local labeled samples used to train the GenMusic VN text emotion/style model.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    script_name = "run_train_text_model.py"
    (kernel_dir / script_name).write_text(_training_kernel_script(dataset_slug=dataset_slug), encoding="utf-8")
    (kernel_dir / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": kernel_ref,
                "title": kernel_slug,
                "code_file": script_name,
                "language": "python",
                "kernel_type": "script",
                "is_private": "true",
                "enable_gpu": "false",
                "enable_internet": "false",
                "machine_shape": "CPU",
                "dataset_sources": [dataset_ref],
                "competition_sources": [],
                "kernel_sources": [],
                "model_sources": [],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    commands = _commands(dataset_dir, kernel_dir, download_dir, kernel_ref)
    commands.append(f'# Sau khi complete, copy "{download_dir}\\genmusic_text_model.json" vào "{local_model_path}".')
    (job_dir / "run_commands.ps1").write_text("\n".join(commands) + "\n", encoding="utf-8")
    state = {
        "run_id": run_id,
        "job_kind": "text_model_training",
        "status": "staged",
        "created_at": _now(),
        "kaggle_ready": False,
        "backend": "kaggle_text_model_training",
        "dataset_ref": dataset_ref,
        "kernel_ref": kernel_ref,
        "run_dir": str(run_dir),
        "job_dir": str(job_dir),
        "dataset_dir": str(dataset_dir),
        "kernel_dir": str(kernel_dir),
        "download_dir": str(download_dir),
        "state_path": str(job_dir / "job_state.json"),
        "training_data_path": str(training_jsonl),
        "local_model_path": str(local_model_path),
        "sample_count": sample_count,
        "total_record_count": len(training_records),
        "seed": seed,
        "commands": commands,
        "messages": ["Kaggle text model training files prepared."],
        "history": [],
        "downloaded_files": [],
        "last_error": "",
        "model_path": "",
        "training_report_path": "",
    }
    _write_state(state)
    return state


def submit_training_kaggle_job(
    state: dict[str, Any],
    *,
    wait: bool,
    poll_seconds: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    cli = kaggle_cli_command()
    if cli is None:
        state["status"] = "needs_setup"
        state["messages"].append("Kaggle CLI was not found.")
        _write_state(state)
        return state

    dataset = _run(cli + ["datasets", "create", "-p", state["dataset_dir"], "-r", "zip"], timeout=600)
    state["history"].append(_history_item("datasets create", dataset))
    if dataset["returncode"] != 0:
        return _fail_or_quota(state, dataset, "Dataset upload failed.")

    state["status"] = "dataset_uploaded"
    state["messages"].append("Training dataset uploaded to Kaggle.")
    _write_state(state)
    if not _wait_for_dataset_ready(state, cli):
        _write_state(state)
        return state

    pushed = _run(cli + ["kernels", "push", "-p", state["kernel_dir"]], timeout=600)
    state["history"].append(_history_item("kernels push", pushed))
    if pushed["returncode"] != 0:
        return _fail_or_quota(state, pushed, "Training kernel submit failed.")

    state["status"] = "submitted"
    state["submitted_at"] = _now()
    state["messages"].append("Kaggle text model training kernel submitted.")
    _write_state(state)
    if wait:
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            state = refresh_text_model_training_job(state)
            if state["status"] in {"complete", "failed", "quota_exhausted"}:
                return state
            time.sleep(max(5, poll_seconds))
        state["status"] = "timeout"
        state["messages"].append("Timed out while waiting for Kaggle text model training.")
        _write_state(state)
    return state


def refresh_text_model_training_job(state_or_path: dict[str, Any] | str | Path) -> dict[str, Any]:
    state = _load_training_state(state_or_path)
    cli = kaggle_cli_command()
    if cli is None:
        state["status"] = "needs_setup"
        state["messages"].append("Kaggle CLI was not found.")
        _write_state(state)
        return state

    status = _run(cli + ["kernels", "status", state["kernel_ref"]], timeout=120)
    state["history"].append(_history_item("kernels status", status))
    text = f"{status['stdout']}\n{status['stderr']}".lower()
    state["last_status_output"] = status["stdout"] or status["stderr"]
    if status["returncode"] != 0:
        return _fail_or_quota(state, status, "Could not read Kaggle training kernel status.")
    if any(marker in text for marker in ["complete", "completed", "succeeded"]):
        state["status"] = "complete"
        _download_training_output(state, cli)
    elif any(marker in text for marker in ["error", "failed", "cancelled", "canceled"]):
        state["status"] = "failed"
        _download_training_output(state, cli)
        if not state.get("last_error"):
            state["last_error"] = state.get("last_status_output", "")
    elif "running" in text:
        state["status"] = "running"
    else:
        state["status"] = "submitted"
    state["checked_at"] = _now()
    _write_state(state)
    return state


def _download_training_output(state: dict[str, Any], cli: list[str]) -> None:
    download_dir = Path(state["download_dir"])
    download_dir.mkdir(parents=True, exist_ok=True)
    output = _run(cli + ["kernels", "output", state["kernel_ref"], "-p", str(download_dir)], timeout=1800)
    state["history"].append(_history_item("kernels output", output))
    files = [path for path in sorted(download_dir.rglob("*")) if path.is_file()]
    state["downloaded_files"] = [str(path) for path in files]
    model_files = [path for path in files if path.name == "genmusic_text_model.json"]
    report_files = [path for path in files if path.name == "training_report.json"]
    if report_files:
        state["training_report_path"] = str(report_files[0])
    if model_files:
        model_path = model_files[0]
        state["model_path"] = str(model_path)
        local_path = Path(state.get("local_model_path") or DEFAULT_LOCAL_MODEL_PATH)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(model_path, local_path)
        state["local_model_path"] = str(local_path)
        state["messages"].append(f"Trained text model downloaded to {local_path}.")
        return
    if state.get("status") == "complete":
        state["status"] = "failed"
    state["last_error"] = _summarize_cli_error(output) or "Kaggle output downloaded, but no genmusic_text_model.json was found."
    state["messages"].append("Kaggle training output downloaded, but no model artifact was found.")


def _training_kernel_script(dataset_slug: str) -> str:
    return f'''from __future__ import annotations

import json
import shutil
import sys
import traceback
import zipfile
from pathlib import Path


INPUT_ROOT = Path("/kaggle/input")
DATASET_DIR = INPUT_ROOT / "{dataset_slug}"
SOURCE_DIR = Path("/kaggle/working/genmusic_vn_source")
OUTPUT_DIR = Path("/kaggle/working/genmusic_text_model")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def find_input_file(name: str) -> Path:
    direct = DATASET_DIR / name
    if direct.exists():
        return direct
    matches = sorted(INPUT_ROOT.rglob(name))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not find {{name}} under {{INPUT_ROOT}}")


def prepare_source() -> None:
    source_zip = find_input_file("genmusic_vn_source.zip")
    if SOURCE_DIR.exists():
        shutil.rmtree(SOURCE_DIR)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source_zip) as archive:
        archive.extractall(SOURCE_DIR)
    sys.path.insert(0, str(SOURCE_DIR))


def main() -> None:
    try:
        prepare_source()
        from genmusic_vn.trained_text_model import train_text_model, write_text_model
        from genmusic_vn.training_dataset import load_training_records

        training_data = find_input_file("training_data.jsonl")
        request_path = find_input_file("training_request.json")
        request = json.loads(request_path.read_text(encoding="utf-8"))
        records = load_training_records([training_data])
        model, report = train_text_model(records, seed=int(request.get("seed", 42)))
        model_path = OUTPUT_DIR / "genmusic_text_model.json"
        report_path = OUTPUT_DIR / "training_report.json"
        write_text_model(model, model_path)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        (OUTPUT_DIR / "training_request.json").write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({{"status": "complete", "model_path": str(model_path), "report": report}}, ensure_ascii=False, indent=2))
    except Exception as exc:
        error = {{"status": "failed", "error": str(exc), "traceback": traceback.format_exc()}}
        (OUTPUT_DIR / "training_error.json").write_text(json.dumps(error, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(error, ensure_ascii=False, indent=2))
        raise


main()
'''


def _fail_or_quota(state: dict[str, Any], result: dict[str, Any], message: str) -> dict[str, Any]:
    error = _summarize_cli_error(result)
    state["last_error"] = error
    if _is_quota_error(error):
        state["status"] = "quota_exhausted"
        state["messages"].append(f"{message} Kaggle quota/capacity appears exhausted.")
    else:
        state["status"] = "failed"
        state["messages"].append(message)
    _write_state(state)
    return state


def _is_quota_error(value: str) -> bool:
    lowered = value.lower()
    return any(marker in lowered for marker in QUOTA_MARKERS)


def _load_training_state(state_or_path: dict[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(state_or_path, dict):
        return state_or_path
    return json.loads(Path(state_or_path).read_text(encoding="utf-8"))
