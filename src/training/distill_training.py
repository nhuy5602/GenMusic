"""Knowledge distillation from the real DiffRhythm2 teacher into the MicroDiT student.

This implementation replicates the *exact* call contract of the official teacher
(`diffrhythm2.backbones.dit.DiT.forward`, as used in `diffrhythm2/cfm.py`'s
`sample_block_cache`), instead of guessing architecture dimensions or fabricating
inputs. See docs/experiments/distillation_fix.md for the reverse-engineering
notes this is based on. In short, the teacher's DiT processes lyric tokens and
the noisy mel latent as ONE shared sequence (lyric tokens get `time=-1` as a
sentinel, noisy frames get their real flow-matching `t`); style conditioning is
a single 512-dim MuQ-MuLan embedding added at every position. We replicate that
by concatenating [text_tokens; noisy_latent] and running one non-cached forward
pass (the KV-cache in the original is a streaming/perf optimization only, not a
semantic difference from a single full-context forward pass).

If the teacher (or its lyric tokenizer) cannot be loaded -- e.g. no internet, or
the DiffRhythm2 repo isn't vendored on PYTHONPATH -- this trains the student on
the ground-truth CFM loss alone and says so plainly in the report, rather than
silently distilling against a randomly-initialized stand-in (the previous
"DummyTeacher" behavior, which produced a meaningless training signal with no
error surfaced to the user).
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from src.models.text_to_music_diffusion import MusicDiffusionConfig
from src.models.dit_transformer import MicroDiT
from src.training.self_diffusion import MusicDiffusionDataset, _torch

TEACHER_COND_DIM = 512


def _load_teacher(repo_id: str, teacher_checkpoint_path: str | Path | None, device: str) -> tuple[nn.Module | None, dict | None, str]:
    """Downloads/loads the real DiffRhythm2 DiT backbone with its own config.json
    dimensions (not guessed). Returns (module_or_None, model_config_or_None, status_message).
    """
    try:
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
    except ImportError as exc:
        return None, None, f"huggingface_hub/safetensors not installed: {exc}"

    try:
        from diffrhythm2.backbones.dit import DiT
    except ImportError as exc:
        return None, None, (
            f"diffrhythm2 package not importable ({exc}). The DiffRhythm2 repo must be cloned and "
            "added to PYTHONPATH (see scripts/run_kaggle_distill.py) -- this only works on Kaggle."
        )

    try:
        config_path = hf_hub_download(repo_id=repo_id, filename="config.json", local_dir="./ckpt")
        with open(config_path) as f:
            model_config = json.load(f)
        model_config["use_flex_attn"] = False

        teacher = DiT(**model_config)

        if teacher_checkpoint_path is not None and Path(teacher_checkpoint_path).exists():
            ckpt_path = Path(teacher_checkpoint_path)
        else:
            ckpt_path = Path(hf_hub_download(repo_id=repo_id, filename="model.safetensors", local_dir="./ckpt"))

        if ckpt_path.name.endswith(".safetensors"):
            payload = load_file(str(ckpt_path))
        else:
            payload = torch.load(ckpt_path, map_location="cpu")
        # Checkpoint is for the full CFM(transformer=DiT(...)) wrapper; the DiT's
        # own weights live under the "transformer." prefix.
        state_dict = payload["model"] if "model" in payload else payload
        stripped = {k.removeprefix("transformer."): v for k, v in state_dict.items() if k.startswith("transformer.")}
        if not stripped:
            stripped = state_dict
        missing, unexpected = teacher.load_state_dict(stripped, strict=False)
        if missing or unexpected:
            print(f"[WARNING] Teacher weight load: {len(missing)} missing, {len(unexpected)} unexpected keys.", flush=True)
        teacher.to(device)
        return teacher, model_config, "ok"
    except Exception as e:
        return None, None, f"failed to load real teacher: {e}"


_STRUCT_INFO = {
    "[start]": 500, "[end]": 501, "[intro]": 502, "[verse]": 503, "[chorus]": 504,
    "[outro]": 505, "[inst]": 506, "[solo]": 507, "[bridge]": 508, "[hook]": 509,
    "[break]": 510, "[stop]": 511, "[space]": 512,
}


def _load_lyric_tokenizer():
    """Ports DiffRhythm2's CNENTokenizer/parse_lyrics logic (from inference.py)
    WITHOUT importing that module. `inference.py`'s top-level imports pull in
    `bigvgan`, whose CUDA extension (`bigvgan/alias_free_activation/cuda/activation1d.py`)
    calls `torch.utils.cpp_extension.load()` -- a JIT compile -- as a bare
    module-level statement executed at *import time*, not lazily. This hung for
    10+ hours in testing on Kaggle (see docs/experiments/distillation_fix.md) and
    has nothing to do with tokenization; only `g2p.g2p_generation.chn_eng_g2p`
    (a lightweight ONNX-based G2P, no CUDA) is actually needed here.

    Note this tokenizer's G2P frontend targets Chinese/English lyrics; Vietnamese
    text will still tokenize deterministically (so it's a valid, if imprecise,
    conditioning signal for the *teacher's* audio prior) but has no linguistic
    grounding for Vietnamese. The student's own text conditioning (frozen
    xlm-roberta-base, which does understand Vietnamese) is what actually
    carries lyric semantics -- the teacher's role in distillation is to transfer
    its general music/audio generation prior, not lyric understanding.
    """
    try:
        import re

        from g2p.g2p_generation import chn_eng_g2p

        struct_pattern = re.compile(r"^\[.*?\]$")

        def parse_lyrics(lyrics: str):
            lyrics_with_time = []
            get_start = False
            for line in lyrics.split("\n"):
                line = line.strip()
                if not line:
                    continue
                if struct_pattern.match(line):
                    struct_idx = _STRUCT_INFO.get(line.lower())
                    if struct_idx is not None:
                        if struct_idx == _STRUCT_INFO["[start]"]:
                            get_start = True
                        lyrics_with_time.append([struct_idx, _STRUCT_INFO["[stop]"]])
                    continue
                _, token = chn_eng_g2p(line)
                lyrics_with_time.append([t + 1 for t in token] + [_STRUCT_INFO["[stop]"]])
            if lyrics_with_time and not get_start:
                lyrics_with_time = [[_STRUCT_INFO["[start]"], _STRUCT_INFO["[stop]"]]] + lyrics_with_time
            return lyrics_with_time

        return parse_lyrics, "ok"
    except Exception as e:
        return None, f"failed to load lyric tokenizer: {e}"


def _tokenize_lyrics_batch(parse_lyrics_fn, texts: list[str], device) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenizes each lyric string with the real teacher tokenizer and
    right-pads (filler token 0, per TextEmbedding's own convention) to a
    common length. Returns (token_ids [B, T], valid_mask [B, T])."""
    token_lists = []
    for text in texts:
        wrapped = f"[start]\n[verse]\n{text.strip()}\n[stop]"
        try:
            grouped = parse_lyrics_fn(wrapped)
            token_lists.append([int(t) for t in sum(grouped, [])] or [0])
        except Exception:
            token_lists.append([0])
    max_len = max(len(t) for t in token_lists)
    batch = len(token_lists)
    ids = torch.zeros(batch, max_len, dtype=torch.long, device=device)
    valid = torch.zeros(batch, max_len, dtype=torch.bool, device=device)
    for i, toks in enumerate(token_lists):
        ids[i, : len(toks)] = torch.tensor(toks, dtype=torch.long, device=device)
        valid[i, : len(toks)] = True
    return ids, valid


class KnowledgeDistillationTrainer:
    """Orchestrates distillation transfer from the real (or unavailable) DiffRhythm2
    teacher to a MicroDiT student."""

    def __init__(
        self,
        teacher_model: nn.Module | None,
        student_model: MicroDiT,
        config: MusicDiffusionConfig,
        optimizer: torch.optim.Optimizer | None,
        device: str = "cpu",
        alpha_feature: float = 0.5,
        parse_lyrics_fn=None,
        teacher_mel_dim: int | None = None,
    ):
        self.teacher = teacher_model.to(device) if teacher_model is not None else None
        self.student = student_model.to(device)
        self.config = config
        self.optimizer = optimizer
        self.device = device
        self.parse_lyrics_fn = parse_lyrics_fn
        # If the teacher (or its tokenizer) is unavailable, there is no
        # distillation signal to blend in -- fall back to pure ground-truth CFM.
        self.alpha_feature = 1.0 if (self.teacher is None or parse_lyrics_fn is None) else alpha_feature

        # The teacher's real checkpoint (mel_dim=64, read from its own config.json)
        # does not match our student's Vocos-native mel space (mel_dim=100, chosen
        # to fix the vocoder distortion bug -- see docs/experiments/vocoder_fix.md).
        # A small trainable linear adapter bridges the two mel-filterbank spaces so
        # distillation can still transfer signal, without touching the student's own
        # generative/decode path. Both are smooth linear-ish remappings of the same
        # underlying STFT magnitude spectrum, so a learned linear map is a reasonable
        # (if imperfect) bridge -- see docs/experiments/distillation_fix.md.
        self.to_teacher_mel = None
        self.from_teacher_mel = None
        if self.teacher is not None and teacher_mel_dim is not None and teacher_mel_dim != config.n_mels:
            self.to_teacher_mel = nn.Linear(config.n_mels, teacher_mel_dim).to(device)
            self.from_teacher_mel = nn.Linear(teacher_mel_dim, config.n_mels).to(device)

        if self.teacher is not None:
            self.teacher.eval()
            for param in self.teacher.parameters():
                param.requires_grad = False

    def adapter_parameters(self) -> list[torch.nn.Parameter]:
        params = []
        if self.to_teacher_mel is not None:
            params += list(self.to_teacher_mel.parameters()) + list(self.from_teacher_mel.parameters())
        return params

    def _teacher_velocity(self, xt: torch.Tensor, t: torch.Tensor, texts: list[str], style_prompt: torch.Tensor) -> torch.Tensor | None:
        if self.teacher is None or self.parse_lyrics_fn is None:
            return None
        batch_size, seq_len = xt.shape[0], xt.shape[1]

        token_ids, token_valid = _tokenize_lyrics_batch(self.parse_lyrics_fn, texts, self.device)
        text_len = token_ids.shape[1]

        text_emb = self.teacher.text_embed(token_ids)  # (B, text_len, cond_dim)
        text_time = torch.full((batch_size, text_len), -1.0, device=self.device, dtype=xt.dtype)
        text_position_ids = torch.arange(text_len, device=self.device).unsqueeze(0).repeat(batch_size, 1)

        teacher_xt = self.to_teacher_mel(xt) if self.to_teacher_mel is not None else xt
        noisy_latent = self.teacher.latent_embed(teacher_xt)  # (B, seq_len, cond_dim)
        noisy_time = t[:, None].repeat(1, seq_len)
        noisy_position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0).repeat(batch_size, 1)

        x = torch.cat([text_emb, noisy_latent], dim=1)
        time = torch.cat([text_time, noisy_time], dim=1)
        position_ids = torch.cat([text_position_ids, noisy_position_ids], dim=1)

        total_len = text_len + seq_len
        key_valid = torch.cat([token_valid, torch.ones(batch_size, seq_len, dtype=torch.bool, device=self.device)], dim=1)
        attn_mask = key_valid[:, None, None, :].repeat(1, 1, total_len, 1)  # (B, 1, total_len, total_len), True=attend

        outputs = self.teacher(
            x=x,
            time=time,
            position_ids=position_ids,
            style_prompt=style_prompt,
            attn_mask=attn_mask,
            use_cache=False,
            past_key_value=None,
        )
        pred = outputs[0] if isinstance(outputs, tuple) else outputs
        teacher_velocity = pred[:, text_len:]
        if self.from_teacher_mel is not None:
            teacher_velocity = self.from_teacher_mel(teacher_velocity)
        return teacher_velocity

    def train_epoch(self, dataloader) -> list[dict[str, float | None]]:
        """Returns per-step {"loss": total, "loss_gt": ground-truth CFM component,
        "loss_velocity": teacher-matching component or None} -- kept separate (not
        just the blended total) so distilled vs. non-distilled runs can be compared
        on the same ground-truth-loss axis. See docs/experiments/*.md."""
        self.student.train()
        epoch_losses = []

        for batch in dataloader:
            vocal_mel = batch["vocal_mel"].to(self.device)  # (B, n_mels, seq_len)
            backing_mel = batch["backing_mel"].to(self.device)
            style_anchor = batch["style_anchor"].to(self.device)  # (B, 512) precomputed MuQ-MuLan embedding
            texts = batch["text"]

            x1 = vocal_mel.transpose(1, 2)  # (B, seq_len, n_mels)
            cond = backing_mel.transpose(1, 2)
            x0 = torch.randn_like(x1)

            batch_size = x1.shape[0]
            t = torch.rand(batch_size, device=self.device)
            xt = (1.0 - t.view(-1, 1, 1)) * x0 + t.view(-1, 1, 1) * x1

            self.optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                v_teacher = self._teacher_velocity(xt, t, texts, style_anchor)

            v_student = self.student(x=xt, cond=cond, texts=texts, timestep=t, style_prompt=style_anchor)

            target_velocity = x1 - x0
            loss_gt = F.mse_loss(v_student, target_velocity)
            loss_velocity = None
            if v_teacher is not None:
                loss_velocity = F.mse_loss(v_student, v_teacher)
                loss = (1.0 - self.alpha_feature) * loss_velocity + self.alpha_feature * loss_gt
            else:
                loss = loss_gt

            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(self.student.parameters()) + self.adapter_parameters(), 1.0)
            self.optimizer.step()

            epoch_losses.append({
                "loss": float(loss.detach().cpu()),
                "loss_gt": float(loss_gt.detach().cpu()),
                "loss_velocity": float(loss_velocity.detach().cpu()) if loss_velocity is not None else None,
            })

        return epoch_losses


def run_distillation_training(
    dataset_dir: str | Path,
    student_checkpoint_path: str | Path,
    teacher_checkpoint_path: str | Path | None = None,
    *,
    epochs: int = 5,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    device: str | None = None,
    alpha_feature: float = 0.5,
    repo_id: str = "ASLP-lab/DiffRhythm2",
    dim: int = 256,
    depth: int = 4,
    heads: int = 4,
    ff_mult: int = 4,
) -> dict[str, Any]:
    torch, _, _, DataLoaderClass = _torch()

    root = Path(dataset_dir)
    student_checkpoint = Path(student_checkpoint_path)
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    config = MusicDiffusionConfig(**json.loads((root / "config.json").read_text(encoding="utf-8")))

    teacher_backbone, teacher_config, teacher_status = _load_teacher(repo_id, teacher_checkpoint_path, selected_device)
    print(f"Teacher load status: {teacher_status}", flush=True)
    teacher_mel_dim = teacher_config.get("mel_dim") if teacher_config else None
    if teacher_backbone is not None and teacher_mel_dim is not None and teacher_mel_dim != config.n_mels:
        print(
            f"Teacher mel_dim={teacher_mel_dim} != dataset n_mels={config.n_mels}; "
            "bridging with a trainable linear adapter (see docs/experiments/distillation_fix.md).",
            flush=True,
        )

    parse_lyrics_fn, tokenizer_status = (None, "skipped (no teacher)") if teacher_backbone is None else _load_lyric_tokenizer()
    if teacher_backbone is not None:
        print(f"Lyric tokenizer status: {tokenizer_status}", flush=True)

    model_student = MicroDiT(config, dim=dim, depth=depth, heads=heads, ff_mult=ff_mult, style_dim=TEACHER_COND_DIM).to(selected_device)

    dataset = MusicDiffusionDataset(root, config)

    def collate_fn(batch):
        vocal_mels = torch.stack([item["vocal_mel"] for item in batch])
        backing_mels = torch.stack([item["backing_mel"] for item in batch])
        style_anchors = torch.stack([item["style_anchor"] for item in batch])
        texts = [item["text"] for item in batch]
        return {"vocal_mel": vocal_mels, "backing_mel": backing_mels, "style_anchor": style_anchors, "text": texts}

    dataloader = DataLoaderClass(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    trainer = KnowledgeDistillationTrainer(
        teacher_model=teacher_backbone,
        student_model=model_student,
        config=config,
        optimizer=None,
        device=selected_device,
        alpha_feature=alpha_feature,
        teacher_mel_dim=teacher_mel_dim,
        parse_lyrics_fn=parse_lyrics_fn,
    )
    trainable_params = [p for p in model_student.parameters() if p.requires_grad] + trainer.adapter_parameters()
    trainer.optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)
    distillation_active = trainer.teacher is not None and trainer.parse_lyrics_fn is not None
    print(
        f"Starting {'distillation' if distillation_active else 'ground-truth-only (teacher unavailable)'} "
        f"training for {epochs} epochs on {selected_device}...",
        flush=True,
    )

    start_time = time.perf_counter()
    losses = []
    loss_curve = []
    for epoch in range(epochs):
        epoch_losses = trainer.train_epoch(dataloader)
        losses.extend(epoch_losses)
        avg_loss = sum(d["loss"] for d in epoch_losses) / len(epoch_losses)
        avg_loss_gt = sum(d["loss_gt"] for d in epoch_losses) / len(epoch_losses)
        velocity_values = [d["loss_velocity"] for d in epoch_losses if d["loss_velocity"] is not None]
        avg_loss_velocity = sum(velocity_values) / len(velocity_values) if velocity_values else None
        loss_curve.append({"epoch": epoch + 1, "loss": avg_loss, "loss_gt": avg_loss_gt, "loss_velocity": avg_loss_velocity})
        print(
            f"Epoch [{epoch+1}/{epochs}] complete. Average Loss: {avg_loss:.6f} "
            f"(loss_gt={avg_loss_gt:.6f}, loss_velocity={avg_loss_velocity})",
            flush=True,
        )

    final_loss = sum(d["loss"] for d in losses[-10:]) / max(1, len(losses[-10:]))
    final_loss_gt = sum(d["loss_gt"] for d in losses[-10:]) / max(1, len(losses[-10:]))

    from src.models.text_to_music_diffusion import save_checkpoint

    save_checkpoint(
        model_student, student_checkpoint, config, optimizer=trainer.optimizer, epoch=epochs, loss=final_loss,
        arch={"dim": dim, "depth": depth, "heads": heads, "ff_mult": ff_mult, "style_dim": TEACHER_COND_DIM},
    )

    report = {
        "status": "complete",
        "backend": "genmusic-vn-dit-distillation",
        "distillation_active": distillation_active,
        "teacher_status": teacher_status,
        "tokenizer_status": tokenizer_status if teacher_backbone is not None else "skipped (no teacher)",
        "teacher_mel_dim": teacher_mel_dim,
        "student_mel_dim": config.n_mels,
        "mel_adapter_used": trainer.to_teacher_mel is not None,
        "student_checkpoint": str(student_checkpoint.resolve()),
        "epochs": epochs,
        "step_count": len(losses),
        "final_loss": round(final_loss, 6),
        "final_loss_gt": round(final_loss_gt, 6),
        "loss_curve": loss_curve,
        "elapsed_seconds": round(time.perf_counter() - start_time, 3),
        "alpha_feature": alpha_feature,
        "dim": dim,
        "depth": depth,
        "heads": heads,
        "ff_mult": ff_mult,
    }

    (student_checkpoint.parent / "distillation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report
