"""Pretrain GenMusic's native vocal recognizer without an external ASR/TTS model.

The recognizer remains a submodule of the final diffusion checkpoint.  This
phase only learns an audio-to-Vietnamese alignment from the dataset's real
vocal stems; it never synthesizes speech and never changes the acoustic DiT.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch

from src.models.native_text import (
    build_frame_text_targets,
    build_timestamped_frame_text_targets,
    ctc_target_text,
    greedy_decode_ctc,
    greedy_decode_frame_text,
    native_ctc_loss,
    native_frame_text_loss,
)
from src.models.text_to_music_diffusion import load_checkpoint, save_checkpoint
from src.models.text_to_music_diffusion import normalize_mel
from src.training.self_diffusion import (
    _fit_mel_frames,
    _filter_training_records,
    _load_mel,
    _read_records,
    clean_vietnamese_lyric,
    lyric_text_for_window,
    split_training_records,
    usable_lyric_spans,
)


class NativeVocalCTCDataset(torch.utils.data.Dataset):
    """Load only the vocal tensor and aligned lyric needed by native CTC."""

    def __init__(
        self,
        root: Path,
        config,
        records: list[dict],
        *,
        deterministic_crop: bool = False,
        crop_seed: int = 5602,
        cache_items: bool = True,
        crops_per_record: int = 1,
    ) -> None:
        self.root = Path(root)
        self.config = config
        self.records = list(records)
        self.deterministic_crop = bool(deterministic_crop)
        self.crop_seed = int(crop_seed)
        self.cache_items = bool(cache_items)
        self.crops_per_record = max(1, int(crops_per_record))
        self._cache: dict[int, dict] = {}

    def __len__(self) -> int:
        return len(self.records) * self.crops_per_record

    def __getitem__(self, index: int) -> dict:
        if index in self._cache:
            return self._cache[index]
        record_index, crop_variant = divmod(index, self.crops_per_record)
        record = self.records[record_index]
        vocal = _load_mel(self.root / record["vocal_mel_path"])
        crop_start = 0
        lyric_spans = usable_lyric_spans(record)
        if vocal.shape[1] > self.config.frames_per_chunk:
            max_start = vocal.shape[1] - self.config.frames_per_chunk
            record_key = str(record.get("id") or record_index)
            if lyric_spans:
                if self.deterministic_crop or self.crops_per_record > 1:
                    digest = hashlib.sha256(
                        f"{self.crop_seed}:{record_key}:lyric:{crop_variant}".encode("utf-8")
                    ).digest()
                    span = lyric_spans[
                        int.from_bytes(digest[:8], "big") % len(lyric_spans)
                    ]
                    fraction = int.from_bytes(digest[8:16], "big") / float(2**64 - 1)
                    focus_seconds = span[0] + fraction * (span[1] - span[0])
                else:
                    span = random.choice(lyric_spans)
                    focus_seconds = random.uniform(span[0], span[1])
                focus_frame = round(
                    focus_seconds * self.config.sample_rate / self.config.hop_length
                )
                crop_start = max(
                    0,
                    min(max_start, focus_frame - self.config.frames_per_chunk // 2),
                )
            elif self.deterministic_crop:
                digest = hashlib.sha256(
                    f"{self.crop_seed}:{record_key}:{crop_variant}".encode("utf-8")
                ).digest()
                crop_start = int.from_bytes(digest[:8], "big") % (max_start + 1)
            else:
                crop_start = random.randint(0, max_start)
            vocal = vocal[:, crop_start:crop_start + self.config.frames_per_chunk]
        else:
            vocal = _fit_mel_frames(vocal, self.config.frames_per_chunk)

        lyric = clean_vietnamese_lyric(str(record["text"]))
        segments = record.get("segments") or []
        if segments:
            crop_start_seconds = (
                crop_start * self.config.hop_length / self.config.sample_rate
            )
            crop_end_seconds = crop_start_seconds + (
                self.config.frames_per_chunk
                * self.config.hop_length
                / self.config.sample_rate
            )
            lyric = lyric_text_for_window(
                lyric,
                segments,
                crop_start_seconds,
                crop_end_seconds,
            )
        crop_start_seconds = crop_start * self.config.hop_length / self.config.sample_rate
        crop_end_seconds = crop_start_seconds + (
            self.config.frames_per_chunk * self.config.hop_length / self.config.sample_rate
        )
        item = {
            "vocal_mel": normalize_mel(vocal, self.config),
            "text": lyric,
            "segments": segments if record.get("exact_word_timestamps") else [],
            "crop_start_seconds": crop_start_seconds,
            "crop_end_seconds": crop_end_seconds,
        }
        if self.cache_items:
            # A fixed 4-second crop is small (~150 KiB) while the source song
            # tensor can be tens of megabytes on Kaggle FUSE. Caching one item
            # per record turns all later CTC epochs into pure GPU work.
            self._cache[index] = item
        return item


def _edit_distance(reference: str, hypothesis: str) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row, ref_char in enumerate(reference, start=1):
        current = [row]
        for column, hyp_char in enumerate(hypothesis, start=1):
            current.append(min(
                current[-1] + 1,
                previous[column] + 1,
                previous[column - 1] + int(ref_char != hyp_char),
            ))
        previous = current
    return previous[-1]


def evaluate_native_text_recognizer(
    model,
    dataloader,
    device: str,
    *,
    objective: str = "ctc",
) -> dict[str, float]:
    """Evaluate real-vocal text loss, sequence CER, and frame accuracy."""
    if objective not in {"ctc", "frame"}:
        raise ValueError(f"Unsupported native text objective: {objective}")
    model.eval()
    losses: list[float] = []
    errors = 0
    character_count = 0
    exact = 0
    sample_count = 0
    correct_frames = 0
    target_frames = 0
    with torch.no_grad():
        for batch in dataloader:
            vocal = batch["vocal_mel"].to(device).transpose(1, 2).float()
            logits = model.audio_text_logits(vocal)
            frame_targets = None
            if objective == "frame" and any(batch.get("segments") or []):
                frame_targets = build_timestamped_frame_text_targets(
                    batch["segments"],
                    crop_starts_seconds=batch["crop_start_seconds"],
                    crop_ends_seconds=batch["crop_end_seconds"],
                    time_steps=logits.shape[1],
                    device=logits.device,
                )
            loss = (
                native_ctc_loss(logits, batch["text"])
                if objective == "ctc"
                else native_frame_text_loss(
                    logits,
                    batch["text"],
                    targets=frame_targets,
                )
            )
            losses.append(float(loss.detach().cpu()))
            if objective == "frame":
                targets = frame_targets
                if targets is None:
                    targets = build_frame_text_targets(
                        batch["text"],
                        time_steps=logits.shape[1],
                        device=logits.device,
                    )
                valid = targets != -100
                correct_frames += int(
                    ((logits.argmax(dim=-1) == targets) & valid).sum().cpu()
                )
                target_frames += int(valid.sum().cpu())
                # Exact timestamp supervision deliberately leaves instrumental
                # and silent frames unlabeled. Decoding those frames would turn
                # harmless unconstrained predictions into CER insertions and
                # reject the checkpoint even while voiced-frame accuracy rises.
                hypotheses = [
                    (
                        greedy_decode_frame_text(
                            logits[index][valid[index]].unsqueeze(0)
                        )[0]
                        if bool(valid[index].any())
                        else ""
                    )
                    for index in range(logits.shape[0])
                ]
            else:
                hypotheses = greedy_decode_ctc(logits)
            for text, hypothesis in zip(batch["text"], hypotheses):
                reference = ctc_target_text(text)
                if objective == "frame":
                    # Silence is ignored and word spaces are not audible frame
                    # labels; compare pronounceable graphemes only.
                    reference = reference.replace(" ", "")
                    hypothesis = hypothesis.replace(" ", "")
                if not reference:
                    continue
                errors += _edit_distance(reference, hypothesis)
                character_count += len(reference)
                exact += int(reference == hypothesis)
                sample_count += 1
    result = {
        "loss": sum(losses) / max(1, len(losses)),
        "cer": errors / max(1, character_count),
        "character_accuracy": max(0.0, 1.0 - errors / max(1, character_count)),
        "exact_match_rate": exact / max(1, sample_count),
        "samples": sample_count,
    }
    if objective == "frame":
        result["frame_accuracy"] = correct_frames / max(1, target_frames)
        result["target_frames"] = target_frames
    return result


def evaluate_native_ctc(model, dataloader, device: str) -> dict[str, float]:
    """Backward-compatible CTC evaluator used by existing tooling."""
    return evaluate_native_text_recognizer(
        model,
        dataloader,
        device,
        objective="ctc",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("dataset_dir")
    parser.add_argument("output_checkpoint")
    parser.add_argument("--report", default="ctc_pretrain_report.json")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--validation-max-records", type=int, default=128)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--minimum-epochs", type=int, default=5)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--crops-per-record", type=int, default=3)
    parser.add_argument(
        "--objective",
        choices=("ctc", "frame"),
        default="ctc",
        help="Train CTC or monotonic frame-level grapheme recognition.",
    )
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if args.epochs < 1:
        raise ValueError("--epochs must be positive")
    if args.learning_rate <= 0.0:
        raise ValueError("--learning-rate must be positive")

    random.seed(5602)
    torch.manual_seed(5602)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, config, payload = load_checkpoint(
        args.checkpoint,
        device=device,
        use_ema=False,
    )
    if not bool(getattr(model, "native_generation", False)):
        raise ValueError("Native CTC pretraining requires a native_utf8 checkpoint")
    recognizer = getattr(model, "audio_text_recognizer", None)
    if recognizer is None:
        raise ValueError("Checkpoint has no native audio-text recognizer")

    dataset_root = Path(args.dataset_dir)
    records = _filter_training_records(_read_records(dataset_root))
    if args.max_records is not None:
        records = records[: max(1, int(args.max_records))]
    train_records, validation_records = split_training_records(
        records,
        validation_fraction=args.validation_fraction,
        validation_max_records=args.validation_max_records,
        seed=5602,
    )
    if not train_records or not validation_records:
        raise ValueError("CTC pretraining requires non-empty train and validation splits")

    train_dataset = NativeVocalCTCDataset(
        dataset_root,
        config,
        train_records,
        crops_per_record=max(1, args.crops_per_record),
    )
    validation_dataset = NativeVocalCTCDataset(
        dataset_root,
        config,
        validation_records,
        deterministic_crop=True,
        crop_seed=5602,
    )

    def collate(batch):
        return {
            "vocal_mel": torch.stack([item["vocal_mel"] for item in batch]),
            "text": [item["text"] for item in batch],
            "segments": [item["segments"] for item in batch],
            "crop_start_seconds": [item["crop_start_seconds"] for item in batch],
            "crop_end_seconds": [item["crop_end_seconds"] for item in batch],
        }

    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=max(1, args.batch_size),
        shuffle=False,
        collate_fn=collate,
    )
    optimizer = torch.optim.AdamW(
        recognizer.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,
    )
    baseline = evaluate_native_text_recognizer(
        model,
        validation_loader,
        device,
        objective=args.objective,
    )
    print(f"{args.objective}_baseline=" + json.dumps(baseline), flush=True)

    best_loss = float(baseline["loss"])
    best_cer = float(baseline["cer"])
    best_epoch = 0
    best_state = {
        name: value.detach().cpu().clone()
        for name, value in recognizer.state_dict().items()
    }
    history: list[dict] = []
    epochs_without_improvement = 0
    for epoch_index in range(args.epochs):
        generator = torch.Generator().manual_seed(5602 + epoch_index)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=max(1, args.batch_size),
            shuffle=True,
            generator=generator,
            collate_fn=collate,
        )
        model.train()
        losses: list[float] = []
        for batch_index, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            vocal = batch["vocal_mel"].to(device).transpose(1, 2).float()
            logits = model.audio_text_logits(vocal)
            frame_targets = None
            if args.objective == "frame" and any(batch.get("segments") or []):
                frame_targets = build_timestamped_frame_text_targets(
                    batch["segments"],
                    crop_starts_seconds=batch["crop_start_seconds"],
                    crop_ends_seconds=batch["crop_end_seconds"],
                    time_steps=logits.shape[1],
                    device=logits.device,
                )
            loss = (
                native_ctc_loss(logits, batch["text"])
                if args.objective == "ctc"
                else native_frame_text_loss(
                    logits,
                    batch["text"],
                    targets=frame_targets,
                )
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(
                    f"Non-finite {args.objective} loss at epoch "
                    f"{epoch_index + 1}, batch {batch_index}"
                )
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                recognizer.parameters(), max_norm=5.0
            )
            if not bool(torch.isfinite(gradient_norm)):
                raise FloatingPointError(
                    f"Non-finite {args.objective} gradient at epoch "
                    f"{epoch_index + 1}, batch {batch_index}"
                )
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            if batch_index % 10 == 0:
                print(
                    f"{args.objective}_epoch={epoch_index + 1}/{args.epochs} "
                    f"batch={batch_index}/{len(train_loader)} loss={losses[-1]:.6f}",
                    flush=True,
                )

        validation = evaluate_native_text_recognizer(
            model,
            validation_loader,
            device,
            objective=args.objective,
        )
        epoch_result = {
            "epoch": epoch_index + 1,
            "train_loss": sum(losses) / max(1, len(losses)),
            "validation": validation,
        }
        history.append(epoch_result)
        print(
            f"{args.objective}_epoch_result=" + json.dumps(epoch_result),
            flush=True,
        )
        validation_cer = float(validation["cer"])
        validation_loss = float(validation["loss"])
        # The recognizer exists to provide phoneme/content gradients, so greedy
        # CER is the primary checkpoint criterion. CTC loss only breaks ties;
        # selecting by loss alone previously discarded a clearly better CER.
        is_improvement = (
            validation_cer < best_cer - 1e-4
            or (
                abs(validation_cer - best_cer) <= 1e-4
                and validation_loss < best_loss - 1e-4
            )
        )
        if is_improvement:
            best_cer = validation_cer
            best_loss = validation_loss
            best_epoch = epoch_index + 1
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in recognizer.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if (
            epoch_index + 1 >= max(1, args.minimum_epochs)
            and epochs_without_improvement >= max(1, args.patience)
        ):
            break

    recognizer.load_state_dict(best_state)
    final_validation = evaluate_native_text_recognizer(
        model,
        validation_loader,
        device,
        objective=args.objective,
    )
    if not math.isfinite(final_validation["loss"]):
        raise FloatingPointError("Final CTC validation loss is non-finite")

    # Inference loads EMA by default. Replace only the recognizer tensors in
    # the existing EMA snapshot so the improved CTC survives the next resume;
    # every acoustic/text parameter remains byte-for-byte unchanged.
    ema_state = {
        name: value.detach().cpu().clone()
        for name, value in (payload.get("ema") or {}).items()
    }
    for name, value in model.state_dict().items():
        if name.startswith("audio_text_recognizer."):
            ema_state[name] = value.detach().cpu().clone()

    previous_training_state = dict(payload.get("training_state") or {})
    state_key = f"native_{args.objective}_text_pretrain"
    previous_training_state[state_key] = {
        "completed_epochs": len(history),
        "baseline": baseline,
        "final": final_validation,
    }
    output_checkpoint = Path(args.output_checkpoint)
    save_checkpoint(
        model,
        output_checkpoint,
        config,
        ema_state=ema_state or None,
        epoch=int(payload.get("epoch", 0)),
        loss=payload.get("loss"),
        arch=payload.get("arch") or {},
        training_state=previous_training_state,
    )
    report = {
        "status": "complete",
        "phase": f"native_{args.objective}_text_pretrain",
        "objective": args.objective,
        "checkpoint": str(output_checkpoint.resolve()),
        "dataset_records": len(records),
        "training_records": len(train_records),
        "training_examples": len(train_dataset),
        "crops_per_record": max(1, args.crops_per_record),
        "validation_records": len(validation_records),
        "baseline": baseline,
        "final": final_validation,
        "best_epoch": best_epoch,
        "history": history,
        "external_tts_or_asr_used": False,
        "acoustic_model_updated": False,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"{args.objective}_pretrain_complete=" + json.dumps(report["final"]),
        flush=True,
    )


if __name__ == "__main__":
    main()
