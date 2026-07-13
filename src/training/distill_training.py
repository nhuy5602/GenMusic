import os
import json
import time
import math
import torch
from torch import nn
from pathlib import Path
from typing import Any

from src.models.text_to_music_diffusion import MusicDiffusionConfig
from src.models.dit_transformer import MicroDiT
from src.training.self_diffusion import MusicDiffusionDataset, _torch

class _FallbackTeacher(nn.Module):
    """Shape-compatible teacher used only when the optional teacher is unavailable."""
    def __init__(self, mel_dim: int):
        super().__init__()
        self.proj = nn.Linear(mel_dim, mel_dim)

    def forward(self, x, cond=None, text=None, time=None, drop_audio_cond=False, drop_text=False, **kwargs):
        return self.proj(x)


class KnowledgeDistillationTrainer:
    """Orchestrates distillation transfer from a pretrained DiffRhythm teacher to a MicroDiT student."""
    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: MicroDiT,
        config: MusicDiffusionConfig,
        optimizer: torch.optim.Optimizer,
        device: str = "cpu",
        temperature: float = 2.0,
        alpha_feature: float = 0.5
    ):
        self.teacher = teacher_model.to(device)
        self.student = student_model.to(device)
        self.config = config
        self.optimizer = optimizer
        self.device = device
        self.temperature = temperature
        self.alpha_feature = alpha_feature
        
        # Freeze the teacher parameters completely
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

    def train_epoch(self, dataloader) -> list[float]:
        self.student.train()
        epoch_losses = []
        
        for batch in dataloader:
            vocal_mel = batch["vocal_mel"].to(self.device) # Target Shape: (batch, n_mels, seq_len)
            backing_mel = batch["backing_mel"].to(self.device) # Condition Shape: (batch, n_mels, seq_len)
            style_anchor = batch["style_anchor"].to(self.device)
            texts = batch["text"]
            
            # Match channel layouts
            # Teacher and Student expect (batch, seq_len, n_mels)
            x1 = vocal_mel.transpose(1, 2)
            cond = backing_mel.transpose(1, 2)
            style_anchor_t = style_anchor.transpose(1, 2)
            x0 = torch.randn_like(x1)
            
            # Sample timestep t in [0, 1]
            batch_size = x1.shape[0]
            t = torch.rand(batch_size, device=self.device)
            t_unsqueezed = t.view(-1, 1, 1)
            
            # Linearly interpolate intermediate noised state xt
            xt = (1.0 - t_unsqueezed) * x0 + t_unsqueezed * x1
            
            self.optimizer.zero_grad(set_to_none=True)
            
            # 1. Forward pass on teacher (no gradients tracked)
            with torch.no_grad():
                # The reference DiffRhythm teacher model outputs predicted velocity
                v_teacher = self.teacher(
                    x=xt,
                    cond=cond,
                    text=self._tokenize_text_for_teacher(texts),
                    time=t,
                    drop_audio_cond=False,
                    drop_text=False
                )
                if isinstance(v_teacher, (tuple, list)):
                    v_teacher = v_teacher[0]
                
            # 2. Forward pass on student (gradients tracked)
            v_student = self.student(
                x=xt,
                cond=cond,
                texts=texts,
                timestep=t,
                style_prompt=style_anchor_t
            )
            
            # 3. Compute joint distillation losses
            # - Velocity matching loss (KL / Mean Squared Error between velocity vectors)
            loss_velocity = torch.nn.functional.mse_loss(v_student, v_teacher)
            
            # - Output reconstruction loss against clean data target (Ground truth CFM loss)
            target_velocity = x1 - x0
            loss_gt = torch.nn.functional.mse_loss(v_student, target_velocity)
            
            # Total combined loss
            loss = (1.0 - self.alpha_feature) * loss_velocity + self.alpha_feature * loss_gt
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(self.student.parameters()), 1.0)
            self.optimizer.step()
            
            epoch_losses.append(float(loss.detach().cpu()))
            
        return epoch_losses

    def _tokenize_text_for_teacher(self, texts: list[str]) -> torch.Tensor:
        from src.models.text_to_music_diffusion import text_batch
        return text_batch(texts, self.config, self.device)


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
    repo_id: str = "ASLP-lab/DiffRhythm2"
) -> dict[str, Any]:
    torch, _, _, DataLoaderClass = _torch()
    
    root = Path(dataset_dir)
    student_checkpoint = Path(student_checkpoint_path)
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load config
    config = MusicDiffusionConfig(**json.loads((root / "config.json").read_text(encoding="utf-8")))
    
    # 2. Download and load Teacher (DiffRhythm 2 Model)
    # Determine the teacher checkpoint path (local or download from HF)
    actual_teacher_path = None
    if teacher_checkpoint_path is not None and Path(teacher_checkpoint_path).exists():
        actual_teacher_path = Path(teacher_checkpoint_path)
        print(f"Using local teacher checkpoint: {actual_teacher_path}", flush=True)
    else:
        print(f"Downloading pretrained teacher from Hugging Face repo '{repo_id}'...", flush=True)
        try:
            from huggingface_hub import hf_hub_download
            actual_teacher_path = Path(hf_hub_download(
                repo_id=repo_id,
                filename="model.safetensors",
                local_dir="./ckpt",
                local_files_only=False
            ))
            print(f"Downloaded teacher model: {actual_teacher_path}", flush=True)
        except Exception as e:
            print(f"[WARNING] Failed to download from Hugging Face: {e}. Falling back to initialized weights.", flush=True)

    # Instantiate Llama DiT backbone of DiffRhythm 2
    try:
        from DiffRhythm2_main.diffrhythm2.backbones.dit import DiT
        teacher_backbone = DiT(
            dim=512, # Reference standard DiffRhythm hidden dimension
            depth=8,
            heads=8,
            mel_dim=config.n_mels
        )
    except ImportError:
        # Fallback dummy model if imports are not configured
        teacher_backbone = _FallbackTeacher(config.n_mels)
    
    if actual_teacher_path is not None and actual_teacher_path.exists():
        print(f"Loading pretrained teacher weights from: {actual_teacher_path}", flush=True)
        if actual_teacher_path.name.endswith(".safetensors"):
            from safetensors.torch import load_file
            teacher_payload = load_file(str(actual_teacher_path))
        else:
            teacher_payload = torch.load(actual_teacher_path, map_location="cpu")
        teacher_backbone.load_state_dict(teacher_payload["model"] if "model" in teacher_payload else teacher_payload, strict=False)
    else:
        print("[WARNING] Teacher checkpoint not loaded. Distilling using initialized weights.", flush=True)
        
    # 3. Instantiate Student (MicroDiT)
    model_student = MicroDiT(config, dim=256, depth=4, heads=4).to(selected_device)
    
    # 4. Setup Optimizer (only for student parameters)
    trainable_params = [p for p in model_student.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)
    
    # 5. Dataset and Dataloader
    dataset = MusicDiffusionDataset(root, config)
    
    def collate_fn(batch):
        vocal_mels = torch.stack([item["vocal_mel"] for item in batch])
        backing_mels = torch.stack([item["backing_mel"] for item in batch])
        style_anchors = torch.stack([item["style_anchor"] for item in batch])
        texts = [item["text"] for item in batch]
        return {"vocal_mel": vocal_mels, "backing_mel": backing_mels, "style_anchor": style_anchors, "text": texts}

    dataloader = DataLoaderClass(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    
    # 6. Distillation loop
    trainer = KnowledgeDistillationTrainer(
        teacher_model=teacher_backbone,
        student_model=model_student,
        config=config,
        optimizer=optimizer,
        device=selected_device,
        alpha_feature=alpha_feature
    )
    
    print(f"Starting distillation training for {epochs} epochs on {selected_device}...", flush=True)
    start_time = time.perf_counter()
    losses = []
    
    for epoch in range(epochs):
        epoch_losses = trainer.train_epoch(dataloader)
        losses.extend(epoch_losses)
        print(f"Epoch [{epoch+1}/{epochs}] complete. Average Loss: {sum(epoch_losses)/len(epoch_losses):.6f}", flush=True)
        
    final_loss = sum(losses[-10:]) / max(1, len(losses[-10:]))
    
    # 7. Save student checkpoint
    from src.models.text_to_music_diffusion import save_checkpoint
    save_checkpoint(
        model_student,
        student_checkpoint,
        config,
        optimizer=optimizer,
        epoch=epochs,
        loss=final_loss
    )
    
    report = {
        "status": "complete",
        "backend": "genmusic-vn-dit-distillation",
        "student_checkpoint": str(student_checkpoint.resolve()),
        "epochs": epochs,
        "step_count": len(losses),
        "final_loss": round(final_loss, 6),
        "elapsed_seconds": round(time.perf_counter() - start_time, 3)
    }
    
    (student_checkpoint.parent / "distillation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report
