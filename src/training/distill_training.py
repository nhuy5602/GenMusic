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
the DiffRhythm2 repo isn't vendored on PYTHONPATH -- `run_distillation_training`
raises immediately rather than either (a) silently distilling against a
randomly-initialized stand-in (the old "DummyTeacher" behavior) or (b) silently
falling back to ground-truth-only training under the `train-distill` name. If you
want ground-truth-only training, call `train-self` instead; `train-distill`
always means a real teacher was actually used, no exceptions.
"""

import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from src.models.text_to_music_diffusion import MusicDiffusionConfig
from src.models.dit_transformer import MicroDiT
from src.training.self_diffusion import (
    MusicDiffusionDataset,
    _filter_training_records,
    _read_records,
    _torch,
    estimate_vocal_mel_stats,
)

TEACHER_COND_DIM = 512


def _resize_mel_bins(mel: torch.Tensor, target_bins: int) -> torch.Tensor:
    """Deterministic resize across the mel-frequency-bin axis (last dim) via linear
    interpolation. Used to bridge the student's mel-filterbank dimension into the
    teacher's, without a trainable layer whose output would feed into the frozen
    teacher's no_grad-scoped forward pass and therefore never receive a gradient
    anyway (see docs/experiments/distillation_fix.md)."""
    batch, seq_len, n_mels = mel.shape
    flattened = mel.reshape(batch * seq_len, 1, n_mels)
    resized = F.interpolate(flattened, size=target_bins, mode="linear", align_corners=False)
    return resized.reshape(batch, seq_len, target_bins)


def _hf_hub_download_with_retry(*, attempts: int = 8, initial_backoff_seconds: float = 5.0, max_backoff_seconds: float = 60.0, **kwargs) -> str:
    """`hf_hub_download` with no retry has a single-network-blip failure mode:
    one transient Hub hiccup burns an entire multi-hour Kaggle job before
    training even starts (observed three times in practice -- see
    docs/PROJECT_REPORT.md §4.10). All three were genuine HF Hub-side HTTP 504s
    (confirmed by reproducing the same 504 from a completely different network,
    where it self-resolved after ~130s once huggingface_hub's own built-in
    retry rode it out) -- not a Kaggle-specific or code problem. A short 3x5s
    retry budget is not generous enough for that; exponential backoff up to
    `max_backoff_seconds`, doubling each attempt, gives it several minutes to
    recover -- trivial next to the multi-hour job it protects. A genuinely
    missing repo/file still raises after the last attempt.
    """
    from huggingface_hub import hf_hub_download

    last_error: Exception | None = None
    backoff = initial_backoff_seconds
    for attempt in range(1, attempts + 1):
        try:
            return hf_hub_download(**kwargs)
        except Exception as e:
            last_error = e
            if attempt < attempts:
                print(f"[hf_hub_download] attempt {attempt}/{attempts} failed ({e}); retrying in {backoff:.0f}s...", flush=True)
                time.sleep(backoff)
                backoff = min(backoff * 2, max_backoff_seconds)
    raise last_error


def _load_teacher(repo_id: str, teacher_checkpoint_path: str | Path | None, device: str) -> tuple[nn.Module | None, dict | None, str]:
    """Downloads/loads the real DiffRhythm2 DiT backbone with its own config.json
    dimensions (not guessed). Returns (module_or_None, model_config_or_None, status_message).
    """
    try:
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
        config_path = _hf_hub_download_with_retry(repo_id=repo_id, filename="config.json", local_dir="./ckpt")
        with open(config_path) as f:
            model_config = json.load(f)
        model_config["use_flex_attn"] = False

        teacher = DiT(**model_config)

        if teacher_checkpoint_path is not None and Path(teacher_checkpoint_path).exists():
            ckpt_path = Path(teacher_checkpoint_path)
        else:
            ckpt_path = Path(_hf_hub_download_with_retry(repo_id=repo_id, filename="model.safetensors", local_dir="./ckpt"))

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
        # Bridging student->teacher (to_teacher_mel) uses a fixed deterministic
        # interpolation across the mel-bin axis, NOT a trainable layer: its output
        # feeds directly into the frozen teacher's forward pass, which must stay
        # torch.no_grad()-scoped (backward through the ~1.14B-param teacher every
        # step would be prohibitively expensive) -- so a "trainable" layer here
        # would silently never receive a gradient (see docs/experiments/distillation_fix.md
        # for the bug this replaces). Bridging teacher->student (from_teacher_mel) has
        # no such constraint (it only touches the teacher's already-computed output),
        # so it stays a real trainable linear adapter.
        self.teacher_mel_dim = teacher_mel_dim
        self.needs_mel_resize = self.teacher is not None and teacher_mel_dim is not None and teacher_mel_dim != config.n_mels
        self.from_teacher_mel = nn.Linear(teacher_mel_dim, config.n_mels).to(device) if self.needs_mel_resize else None

        if self.teacher is not None:
            self.teacher.eval()
            for param in self.teacher.parameters():
                param.requires_grad = False

    def adapter_parameters(self) -> list[torch.nn.Parameter]:
        return list(self.from_teacher_mel.parameters()) if self.from_teacher_mel is not None else []

    def _teacher_velocity(self, xt: torch.Tensor, t: torch.Tensor, texts: list[str], style_prompt: torch.Tensor) -> torch.Tensor | None:
        if self.teacher is None or self.parse_lyrics_fn is None:
            return None
        batch_size, seq_len = xt.shape[0], xt.shape[1]

        token_ids, token_valid = _tokenize_lyrics_batch(self.parse_lyrics_fn, texts, self.device)
        text_len = token_ids.shape[1]

        text_emb = self.teacher.text_embed(token_ids)  # (B, text_len, cond_dim)
        text_time = torch.full((batch_size, text_len), -1.0, device=self.device, dtype=xt.dtype)
        text_position_ids = torch.arange(text_len, device=self.device).unsqueeze(0).repeat(batch_size, 1)

        teacher_xt = _resize_mel_bins(xt, self.teacher_mel_dim) if self.needs_mel_resize else xt

        with torch.no_grad():
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
            teacher_velocity = pred[:, text_len:].detach()

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

            # Not wrapped in torch.no_grad() here -- from_teacher_mel (inside
            # _teacher_velocity) is a real trainable adapter and needs gradient
            # tracking; only the frozen teacher's own forward pass is no_grad-scoped.
            v_teacher = self._teacher_velocity(xt, t, texts, style_anchor)

            v_student = self.student(x=xt, cond=cond, texts=texts, timestep=t, style_prompt=style_anchor)

            target_velocity = x1 - x0
            # loss_gt: same frame-weighted-MSE + reconstruction + time/frequency-delta
            # formula as cfm_loss() (src/models/cfm_flow.py, train-self path), ported
            # here rather than called directly because distillation needs v_student at
            # the SAME (xt, t) used for the teacher-matching term below -- cfm_loss()
            # samples its own xt/t internally, which would decouple the two losses onto
            # different points. The frame reweighting keeps each frame's own pointwise
            # minimizer at the true conditional velocity (still an MSE per frame, just a
            # non-uniform average across frames), so it doesn't break the marginal-ODE
            # guarantee CFM relies on the way switching to a bare L1 would. Validated
            # on train-self before porting here: this combination raised generated mel
            # std from 1.09 to 3.13 against a real-vocal target of 2.95 (see
            # docs/PROJECT_REPORT.md §4.10/§5).
            frame_energy = x1.mean(dim=-1)
            activity_threshold = torch.quantile(frame_energy.detach(), 0.55, dim=1, keepdim=True)
            activity = torch.sigmoid((frame_energy - activity_threshold) * 2.0)
            frame_weights = (1.0 + 2.0 * activity).unsqueeze(-1)
            frame_weights = frame_weights / frame_weights.mean(dim=(1, 2), keepdim=True).clamp_min(1e-6)
            velocity_loss = ((v_student - target_velocity).square() * frame_weights).mean()
            predicted_clean = xt + (1.0 - t.view(-1, 1, 1)) * v_student
            reconstruction_loss = ((predicted_clean - x1).abs() * frame_weights).mean()
            time_delta_loss = F.l1_loss(torch.diff(predicted_clean, dim=1), torch.diff(x1, dim=1))
            frequency_delta_loss = F.l1_loss(torch.diff(predicted_clean, dim=2), torch.diff(x1, dim=2))
            loss_gt = velocity_loss + 0.15 * reconstruction_loss + 0.05 * (time_delta_loss + frequency_delta_loss)
            loss_velocity = None
            if v_teacher is not None:
                # loss_velocity is a distillation feature-matching term against a single
                # teacher output, not a marginal-expectation target, so it has no such
                # requirement. L1 here specifically because pure-MSE feature-matching
                # distillation is documented to cause "distributional averaging" (a
                # blurry, low-variance mean prediction) -- see docs/PROJECT_REPORT.md
                # §4.10 ablation and its cited sources (Dieleman 2024; DMD/ADM papers).
                loss_velocity = F.l1_loss(v_student, v_teacher)
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
    # Auto-calibrate mel_mean/mel_std the same way train-self does (self_diffusion.py's
    # train_model) -- without this, MusicDiffusionDataset applies identity normalization
    # (mel_mean=0, mel_std=1 defaults), leaving the student to fit raw, unnormalized
    # log-mel targets. See docs/PROJECT_REPORT.md §4.10/§5 for why this specifically
    # matters here: it was one of the changes that fixed a measured low-variance
    # ("regression to the mean") output on the train-self side.
    usable_records = _filter_training_records(_read_records(root))
    mel_mean, mel_std = estimate_vocal_mel_stats(root, usable_records)
    config = replace(config, mel_mean=mel_mean, mel_std=mel_std)

    teacher_backbone, teacher_config, teacher_status = _load_teacher(repo_id, teacher_checkpoint_path, selected_device)
    print(f"Teacher load status: {teacher_status}", flush=True)
    if teacher_backbone is None:
        raise RuntimeError(
            f"train-distill requires the real DiffRhythm2 teacher; it failed to load: {teacher_status}. "
            "Use train-self for ground-truth-only training instead -- train-distill never silently "
            "falls back to a fake/randomly-initialized stand-in teacher."
        )
    teacher_mel_dim = teacher_config.get("mel_dim") if teacher_config else None
    if teacher_mel_dim is not None and teacher_mel_dim != config.n_mels:
        print(
            f"Teacher mel_dim={teacher_mel_dim} != dataset n_mels={config.n_mels}; "
            "bridging with a trainable linear adapter (see docs/experiments/distillation_fix.md).",
            flush=True,
        )

    parse_lyrics_fn, tokenizer_status = _load_lyric_tokenizer()
    print(f"Lyric tokenizer status: {tokenizer_status}", flush=True)
    if parse_lyrics_fn is None:
        raise RuntimeError(f"train-distill requires the real lyric tokenizer; it failed to load: {tokenizer_status}.")

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
    # Always true here -- the raises above guarantee a real teacher + tokenizer.
    distillation_active = True
    print(f"Starting distillation training for {epochs} epochs on {selected_device}...", flush=True)

    start_time = time.perf_counter()
    losses = []
    loss_curve = []
    for epoch in range(epochs):
        epoch_losses = trainer.train_epoch(dataloader)
        losses.extend(epoch_losses)
        # Variable-length lyric batches make PyTorch's CUDA allocator create many
        # differently-sized blocks; reserved-but-unallocated fragmentation grows
        # epoch over epoch until an allocation that would otherwise fit fails.
        # Releasing the cache back between epochs bounds that growth.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
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
        "tokenizer_status": tokenizer_status,
        "teacher_mel_dim": teacher_mel_dim,
        "student_mel_dim": config.n_mels,
        "mel_adapter_used": trainer.needs_mel_resize,
        "mel_adapter_trainable": trainer.from_teacher_mel is not None,
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
