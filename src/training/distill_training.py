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
            mel = batch["mel"].to(self.device) # Shape: (batch, n_mels, seq_len)
            texts = batch["text"]
            
            # Match channel layouts
            # Teacher and Student expect (batch, seq_len, n_mels)
            x1 = mel.transpose(1, 2)
            x0 = torch.randn_like(x1)
            
            # Sample timestep t in [0, 1]
            batch_size = x1.shape[0]
            t = torch.rand(batch_size, device=self.device)
            t_unsqueezed = t.view(-1, 1, 1)
            
            # Linearly interpolate intermediate noised state xt
            xt = (1.0 - t_unsqueezed) * x0 + t_unsqueezed * x1
            cond = torch.zeros_like(x1)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            # 1. Forward pass on teacher (no gradients tracked)
            with torch.no_grad():
                # The reference DiffRhythm teacher model outputs predicted velocity
                # Depending on refer/dit.py API, we extract the flow velocity
                v_teacher = self.teacher(
                    x=xt,
                    cond=cond,
                    text=self._tokenize_text_for_teacher(texts),
                    time=t,
                    drop_audio_cond=False,
                    drop_text=False
                )
                
            # 2. Forward pass on student (gradients tracked)
            v_student = self.student(
                x=xt,
                cond=cond,
                texts=texts,
                timestep=t
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
        # Placeholder mapping character tokens or llama embeddings for the teacher inputs
        # Real-time implementation maps tokens using the vocab loaded in our training script
        from src.models.text_to_music_diffusion import text_batch
        return text_batch(texts, self.config, self.device)


def run_distillation_training(
    dataset_dir: str | Path,
    student_checkpoint_path: str | Path,
    teacher_checkpoint_path: str | Path,
    *,
    epochs: int = 5,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    device: str | None = None,
    alpha_feature: float = 0.5
) -> dict[str, Any]:
    torch, _, _, DataLoaderClass = _torch()
    
    root = Path(dataset_dir)
    student_checkpoint = Path(student_checkpoint_path)
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load config
    config = MusicDiffusionConfig(**json.loads((root / "config.json").read_text(encoding="utf-8")))
    
    # 2. Instantiate pretrained Teacher (DiffRhythm Model)
    # Using Llama DiT backbone defined in refer/dit.py
    from refer.dit import DiT
    teacher_backbone = DiT(
        dim=512, # Reference standard DiffRhythm hidden dimension
        depth=8,
        heads=8,
        mel_dim=config.n_mels
    )
    
    if Path(teacher_checkpoint_path).exists():
        print(f"Loading pretrained teacher checkpoint: {teacher_checkpoint_path}", flush=True)
        teacher_payload = torch.load(teacher_checkpoint_path, map_location=selected_device)
        teacher_backbone.load_state_dict(teacher_payload["model"] if "model" in teacher_payload else teacher_payload)
    else:
        print("[WARNING] Pretrained teacher checkpoint not found. Distilling using initialized weights.", flush=True)
        
    # 3. Instantiate Student (MicroDiT)
    model_student = MicroDiT(config, dim=256, depth=4, heads=4).to(selected_device)
    
    # 4. Setup Optimizer (only for student parameters)
    trainable_params = [p for p in model_student.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)
    
    # 5. Dataset and Dataloader
    dataset = MusicDiffusionDataset(root, config)
    
    def collate_fn(batch):
        mels = torch.stack([item["mel"] for item in batch])
        texts = [item["text"] for item in batch]
        return {"mel": mels, "text": texts}

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
