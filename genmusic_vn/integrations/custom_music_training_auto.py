from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

from .kaggle_auto import (
    DEFAULT_CUSTOM_MUSIC_MODEL,
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
    make_run_id,
    resolve_kaggle_username,
    slugify,
)
from ..data.music_audio_dataset import PUBLIC_MUSIC_DATASET_LICENSE, PUBLIC_MUSIC_DATASET_REF
from ..models.custom_text_to_music import CUSTOM_CHECKPOINT_FILENAME, CUSTOM_MODEL_ID


DEFAULT_CUSTOM_MODEL_PATH = Path("models/current") / CUSTOM_CHECKPOINT_FILENAME


def stage_custom_music_training_job(
    *,
    output_root: str | Path = "outputs/custom_music_training",
    audio_dataset_ref: str = PUBLIC_MUSIC_DATASET_REF,
    model: str = CUSTOM_MODEL_ID,
    max_files: int = 32,
    max_steps: int = 200,
    audio_seconds: int = 16,
    seed: int = 5602,
    config: KaggleJobConfig | None = None,
) -> dict[str, Any]:
    config = config or KaggleJobConfig(model=model, machine_shape="NvidiaTeslaT4")
    run_id = make_run_id(f"custom-music-training-{seed}-{max_files}-{max_steps}")
    username = resolve_kaggle_username(config.username) or "YOUR_KAGGLE_USERNAME"
    run_dir = Path(output_root) / run_id
    job_dir = run_dir / "kaggle_train_custom_music"
    dataset_dir = job_dir / "dataset"
    kernel_dir = job_dir / "kernel"
    download_dir = job_dir / "downloaded_output"
    for path in (dataset_dir, kernel_dir, download_dir):
        path.mkdir(parents=True, exist_ok=True)
    private_slug = slugify(f"genmusic-vn-custom-music-data-{run_id}", max_length=48)
    kernel_slug = slugify(f"genmusic-vn-custom-music-train-{run_id}", max_length=48)
    private_ref = f"{username}/{private_slug}"
    kernel_ref = f"{username}/{kernel_slug}"
    request = {
        "run_id": run_id,
        "job_kind": "custom_music_training",
        "model": model,
        "audio_dataset_ref": audio_dataset_ref,
        "audio_dataset_license": PUBLIC_MUSIC_DATASET_LICENSE,
        "max_files": max(1, int(max_files)),
        "max_steps": max(1, int(max_steps)),
        "audio_seconds": max(4, min(30, int(audio_seconds))),
        "seed": int(seed),
        "created_at": _now(),
    }
    (run_dir / "training_request.json").write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")
    (dataset_dir / "training_request.json").write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_source_zip(dataset_dir / "genmusic_vn_source.zip")
    (dataset_dir / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": private_slug,
                "id": private_ref,
                "licenses": [{"name": "other"}],
                "subtitle": "Demo train model text-to-music tự triển khai của GenMusic VN.",
                "description": "Source và request cho custom Transformer; audio đọc từ dataset CC0 gắn kèm kernel.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    script_name = "run_train_custom_music.py"
    (kernel_dir / script_name).write_text(_custom_training_kernel_script(), encoding="utf-8")
    (kernel_dir / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": kernel_ref,
                "title": kernel_slug,
                "code_file": script_name,
                "language": "python",
                "kernel_type": "script",
                "is_private": "true",
                "enable_gpu": "true",
                "enable_internet": "true",
                "machine_shape": config.machine_shape,
                "dataset_sources": [private_ref, audio_dataset_ref],
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
    (job_dir / "run_commands.ps1").write_text("\n".join(commands) + "\n", encoding="utf-8")
    state = {
        "run_id": run_id,
        "job_kind": "custom_music_training",
        "status": "staged",
        "created_at": _now(),
        "kaggle_ready": False,
        "backend": "custom_text_to_music_transformer",
        "model": model,
        "audio_dataset_ref": audio_dataset_ref,
        "audio_dataset_license": PUBLIC_MUSIC_DATASET_LICENSE,
        "dataset_ref": private_ref,
        "kernel_ref": kernel_ref,
        "run_dir": str(run_dir),
        "job_dir": str(job_dir),
        "dataset_dir": str(dataset_dir),
        "kernel_dir": str(kernel_dir),
        "download_dir": str(download_dir),
        "state_path": str(job_dir / "job_state.json"),
        "local_model_path": str(DEFAULT_CUSTOM_MODEL_PATH),
        "max_files": request["max_files"],
        "max_steps": request["max_steps"],
        "commands": commands,
        "messages": ["Đã chuẩn bị kernel Kaggle train model text-to-music tự code và sinh plot."],
        "history": [],
        "downloaded_files": [],
        "plot_files": [],
        "model_path": "",
        "training_report_path": "",
        "last_error": "",
    }
    _write_state(state)
    return state


def submit_custom_music_training_job(
    *,
    output_root: str | Path = "outputs/custom_music_training",
    audio_dataset_ref: str = PUBLIC_MUSIC_DATASET_REF,
    model: str = CUSTOM_MODEL_ID,
    max_files: int = 32,
    max_steps: int = 200,
    audio_seconds: int = 16,
    seed: int = 5602,
    config: KaggleJobConfig | None = None,
) -> dict[str, Any]:
    config = config or KaggleJobConfig(model=model, machine_shape="NvidiaTeslaT4")
    state = stage_custom_music_training_job(
        output_root=output_root,
        audio_dataset_ref=audio_dataset_ref,
        model=model,
        max_files=max_files,
        max_steps=max_steps,
        audio_seconds=audio_seconds,
        seed=seed,
        config=config,
    )
    if not config.submit:
        state["messages"].append("Chỉ chuẩn bị local; chưa submit kernel Kaggle.")
        _write_state(state)
        return state
    cli = kaggle_cli_command()
    if cli is None:
        state["status"] = "needs_setup"
        state["last_error"] = "Không tìm thấy Kaggle CLI."
        _write_state(state)
        return state
    created = _run(cli + ["datasets", "create", "-p", state["dataset_dir"], "-r", "zip"], timeout=600)
    state["history"].append(_history_item("datasets create", created))
    if created["returncode"] != 0:
        state["status"] = "failed"
        state["last_error"] = _summarize_cli_error(created)
        _write_state(state)
        return state
    state["status"] = "dataset_uploaded"
    _write_state(state)
    if not _wait_for_dataset_ready(state, cli):
        return state
    pushed = _run(cli + ["kernels", "push", "-p", state["kernel_dir"]], timeout=600)
    state["history"].append(_history_item("kernels push", pushed))
    if pushed["returncode"] != 0:
        state["status"] = "failed"
        state["last_error"] = _summarize_cli_error(pushed)
        _write_state(state)
        return state
    state["status"] = "submitted"
    state["submitted_at"] = _now()
    state["messages"].append("Đã submit kernel train custom text-to-music lên Kaggle.")
    _write_state(state)
    if config.wait:
        deadline = time.time() + config.timeout_seconds
        while time.time() < deadline:
            state = refresh_custom_music_training_job(state)
            if state["status"] in {"complete", "failed"}:
                return state
            time.sleep(max(5, config.poll_seconds))
        state["status"] = "timeout"
        _write_state(state)
    return state


def refresh_custom_music_training_job(state_or_path: dict[str, Any] | str | Path) -> dict[str, Any]:
    state = state_or_path if isinstance(state_or_path, dict) else json.loads(Path(state_or_path).read_text(encoding="utf-8"))
    cli = kaggle_cli_command()
    if cli is None:
        state["status"] = "needs_setup"
        _write_state(state)
        return state
    status = _run(cli + ["kernels", "status", state["kernel_ref"]], timeout=120)
    state["history"].append(_history_item("kernels status", status))
    text = f"{status['stdout']}\n{status['stderr']}".lower()
    if status["returncode"] != 0:
        state["status"] = "failed"
        state["last_error"] = _summarize_cli_error(status)
    elif any(marker in text for marker in ("complete", "completed", "succeeded")):
        state["status"] = "complete"
        _download_custom_training_output(state, cli)
    elif any(marker in text for marker in ("error", "failed", "cancelled", "canceled")):
        state["status"] = "failed"
        _download_custom_training_output(state, cli)
    elif "running" in text:
        state["status"] = "running"
    else:
        state["status"] = "submitted"
    state["checked_at"] = _now()
    _write_state(state)
    return state


def _download_custom_training_output(state: dict[str, Any], cli: list[str]) -> None:
    download_dir = Path(state["download_dir"])
    download_dir.mkdir(parents=True, exist_ok=True)
    output = _run(cli + ["kernels", "output", state["kernel_ref"], "-p", str(download_dir)], timeout=1800)
    state["history"].append(_history_item("kernels output", output))
    files = [path for path in sorted(download_dir.rglob("*")) if path.is_file()]
    state["downloaded_files"] = [str(path) for path in files]
    model_files = [path for path in files if path.name == CUSTOM_CHECKPOINT_FILENAME]
    report_files = [path for path in files if path.name == "custom_music_training_report.json"]
    state["plot_files"] = [str(path) for path in files if path.suffix.lower() in {".png", ".json"} and ("plot" in path.name.lower() or "plots" in {part.lower() for part in path.parts})]
    if report_files:
        state["training_report_path"] = str(report_files[0])
    if model_files:
        state["model_path"] = str(model_files[0])
        local_path = Path(state.get("local_model_path") or DEFAULT_CUSTOM_MODEL_PATH)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(model_files[0], local_path)
        state["local_model_path"] = str(local_path)
        state["messages"].append(f"Đã đồng bộ custom model về {local_path}.")
    elif state["status"] == "complete":
        state["status"] = "failed"
        state["last_error"] = _summarize_cli_error(output) or f"Không tìm thấy {CUSTOM_CHECKPOINT_FILENAME}."


def _custom_training_kernel_script() -> str:
    return r'''from __future__ import annotations

import json
import random
import shutil
import subprocess
import sys
import traceback
import zipfile
from pathlib import Path

INPUT_ROOT = Path("/kaggle/input")
SOURCE_DIR = Path("/kaggle/working/genmusic_vn_source")
OUTPUT_DIR = Path("/kaggle/working/custom_music_training")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def ensure(import_name: str, *packages: str) -> None:
    try:
        __import__(import_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", *packages])


def find_file(name: str) -> Path:
    values = sorted(INPUT_ROOT.rglob(name))
    if not values:
        raise FileNotFoundError(f"Không tìm thấy {name} trong {INPUT_ROOT}")
    return values[0]


def prepare_source() -> None:
    source_zips = sorted(INPUT_ROOT.rglob("genmusic_vn_source.zip"))
    if not source_zips:
        extracted = [
            path.parent
            for path in sorted(INPUT_ROOT.rglob("pyproject.toml"))
            if (path.parent / "genmusic_vn").is_dir()
        ]
        if extracted:
            sys.path.insert(0, str(extracted[0]))
            return
        raise FileNotFoundError(f"Không tìm thấy source zip hoặc source đã giải nén trong {INPUT_ROOT}")
    source_zip = source_zips[0]
    if SOURCE_DIR.exists():
        shutil.rmtree(SOURCE_DIR)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source_zip) as archive:
        archive.extractall(SOURCE_DIR)
    sys.path.insert(0, str(SOURCE_DIR))


def caption(features) -> str:
    import numpy as np

    values = np.asarray(features, dtype="float32")
    energy = float(np.mean(values[:, 1]) / 7.0)
    brightness = float(np.mean(values[:, 3]) / 7.0)
    mood = "buồn trầm lắng" if energy < 0.32 else ("sôi động" if energy > 0.62 else "ấm áp cân bằng")
    dynamic = "nhẹ nhàng" if energy < 0.32 else ("mạnh mẽ" if energy > 0.62 else "có nhịp điệu")
    timbre = "sáng trong" if brightness > 0.58 else ("tối sâu" if brightness < 0.30 else "mộc ấm")
    return f"nhạc Việt Nam {mood}, {dynamic}, âm sắc {timbre}, giai điệu giàu cảm xúc, hòa âm sạch"


def main() -> None:
    try:
        prepare_source()
        ensure("torch", "torch")
        ensure("librosa", "librosa==0.10.2.post1", "soundfile")
        ensure("numpy", "numpy")
        ensure("matplotlib", "matplotlib")
        import numpy as np
        import torch
        from genmusic_vn.evaluation.custom_music_metrics import (
            audio_quality_metrics,
            build_custom_music_metric_report,
            write_custom_music_metric_plots,
        )
        from genmusic_vn.models.custom_text_to_music import (
            CUSTOM_MODEL_ID,
            MusicFeatureCodec,
            TextVocabulary,
            create_custom_model,
            render_generated_features,
            save_custom_checkpoint,
        )

        request = json.loads(find_file("training_request.json").read_text(encoding="utf-8"))
        random.seed(int(request.get("seed", 5602)))
        audio_files = [path for path in sorted(INPUT_ROOT.rglob("*.mp3")) if path.is_file()][: int(request.get("max_files", 32))]
        if not audio_files:
            raise FileNotFoundError("Dataset không có MP3 để train model tự code.")
        codec = MusicFeatureCodec(frames=min(120, max(16, int(request.get("audio_seconds", 16) / 0.25))))
        records = []
        for path in audio_files:
            features = np.asarray(codec.extract(path, seconds=int(request.get("audio_seconds", 16))), dtype="int64")
            records.append({"path": path, "features": features, "caption": caption(features)})
        if len(records) < 2:
            raise ValueError("Cần ít nhất 2 audio hợp lệ để tách train/holdout.")
        vocabulary = TextVocabulary.build([record["caption"] for record in records], max_size=8192)
        holdout_count = max(1, len(records) // 5)
        train_records = records[:-holdout_count]
        eval_records = records[-holdout_count:]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = create_custom_model(len(vocabulary.tokens), d_model=192, nhead=6, layers=4, max_text=64).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
        losses = []
        max_steps = max(1, int(request.get("max_steps", 200)))
        for step in range(max_steps):
            record = train_records[step % len(train_records)]
            features = record["features"]
            text_ids = torch.tensor([vocabulary.encode(record["caption"])], dtype=torch.long, device=device)
            target = torch.tensor(features[None, :, :], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(text_ids, target)
            loss = sum(torch.nn.functional.cross_entropy(head.reshape(-1, head.shape[-1]), target[:, :, feature_index].reshape(-1)) for feature_index, head in enumerate(logits))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()), 1.0)
            optimizer.step()
            loss_value = float(loss.detach().cpu())
            losses.append({"step": step + 1, "loss": loss_value, "file": record["path"].name})
            if (step + 1) % 10 == 0:
                print(json.dumps(losses[-1], ensure_ascii=False))

        checkpoint_path = OUTPUT_DIR / "custom_text_to_music.pt"
        save_custom_checkpoint(checkpoint_path, model, vocabulary, codec)
        model.eval()
        correct = [0, 0, 0, 0]
        total = [0, 0, 0, 0]
        with torch.inference_mode():
            for record in eval_records:
                target = torch.tensor(record["features"][None, :, :], dtype=torch.long, device=device)
                text_ids = torch.tensor([vocabulary.encode(record["caption"])], dtype=torch.long, device=device)
                logits = model(text_ids, target)
                for feature_index, head in enumerate(logits):
                    predicted = head.argmax(dim=-1)
                    correct[feature_index] += int((predicted == target[:, :, feature_index]).sum().cpu())
                    total[feature_index] += int(target.shape[1])
        eval_accuracy = {
            "pitch_class": round(correct[0] / max(1, total[0]), 4),
            "energy": round(correct[1] / max(1, total[1]), 4),
            "bass": round(correct[2] / max(1, total[2]), 4),
            "brightness": round(correct[3] / max(1, total[3]), 4),
        }
        prompt_ids = torch.tensor([vocabulary.encode(eval_records[0]["caption"])], dtype=torch.long, device=device)
        with torch.inference_mode():
            generated = model.generate(prompt_ids, max_frames=codec.frames, temperature=0.85)
        generated_features = generated[0].detach().cpu().tolist()
        wav_path = OUTPUT_DIR / "custom_music_after_train.wav"
        render_generated_features(generated_features, wav_path)
        import wave
        with wave.open(str(wav_path), "rb") as audio_handle:
            sample_rate = audio_handle.getframerate()
            audio_data = np.frombuffer(audio_handle.readframes(audio_handle.getnframes()), dtype="<i2").astype("float32") / 32767.0
        metrics = audio_quality_metrics(audio_data, sample_rate)
        metrics["id"] = "custom_after_train"
        report = build_custom_music_metric_report(
            [metrics], model_name=CUSTOM_MODEL_ID, dataset_ref=str(request.get("audio_dataset_ref") or ""), training=True
        )
        report["loss_history"] = losses
        report["dataset_file_count"] = len(records)
        report["train_file_count"] = len(train_records)
        report["holdout_file_count"] = len(eval_records)
        report["holdout_feature_accuracy"] = eval_accuracy
        report["device"] = device
        report["architecture"] = "TextVocabulary -> TransformerEncoder -> TransformerDecoder -> 4 discrete audio-feature heads -> custom renderer"
        report["training_note"] = "Model tự code; demo học đặc trưng audio rời rạc từ MP3, không dùng checkpoint của bên thứ ba."
        report["plots"] = write_custom_music_metric_plots(report, OUTPUT_DIR / "plots")
        (OUTPUT_DIR / "custom_music_training_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"status": "complete", "checkpoint": str(checkpoint_path), "report": report}, ensure_ascii=False, indent=2))
    except Exception as exc:
        error = {"status": "failed", "error": str(exc), "traceback": traceback.format_exc()}
        (OUTPUT_DIR / "custom_music_training_error.json").write_text(json.dumps(error, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(error, ensure_ascii=False, indent=2))
        raise


main()
'''
