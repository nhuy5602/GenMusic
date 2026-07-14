import torch
from torch import nn
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRotaryEmbedding
from transformers.models.llama import LlamaConfig

from .text_to_music_diffusion import MusicDiffusionConfig

class PretrainedRobertaEncoder(nn.Module):
    """Frozen pretrained RoBERTa text encoder to extract rich semantic sequence embeddings."""
    def __init__(self, model_name: str = "xlm-roberta-base", out_dim: int = 256):
        super().__init__()
        print(f"Loading pretrained RoBERTa text encoder: {model_name}...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.roberta = AutoModel.from_pretrained(model_name)
        
        # Freeze all RoBERTa parameters
        for param in self.roberta.parameters():
            param.requires_grad = False
            
        self.projection = nn.Sequential(
            nn.Linear(self.roberta.config.hidden_size, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, texts: list[str], device) -> tuple[torch.Tensor, torch.Tensor]:
        # Tokenize inputs
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        
        # Extract embeddings from RoBERTa (no gradients computed)
        with torch.no_grad():
            outputs = self.roberta(**inputs)
            # Use sequence output (batch_size, seq_len, hidden_size)
            seq_embeddings = outputs.last_hidden_state
            
        # Project to target dimension
        projected = self.projection(seq_embeddings)
        attention_mask = inputs["attention_mask"].bool() # (batch_size, seq_len)
        return projected, attention_mask


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding generator."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        device = timestep.device
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = timestep.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return self.mlp(emb)


class InputEmbedding(nn.Module):
    """Mel, Text, Time, and Style projection block."""
    def __init__(self, mel_dim: int, text_dim: int, out_dim: int, cond_dim: int):
        super().__init__()
        # Project mel (x), mel_cond (cond), text_embed, and time/style embeddings
        self.proj = nn.Linear(mel_dim * 2 + text_dim + cond_dim * 2, out_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, text_embed: torch.Tensor, time_emb: torch.Tensor, style_emb: torch.Tensor) -> torch.Tensor:
        # Expand time and style across seq dimension
        seq_len = x.shape[1]
        time_emb_expanded = time_emb.unsqueeze(1).repeat(1, seq_len, 1)
        style_emb_expanded = style_emb.unsqueeze(1).repeat(1, seq_len, 1)
        
        # Concatenate features and project
        merged = torch.cat((x, cond, text_embed, time_emb_expanded, style_emb_expanded), dim=-1)
        return self.proj(merged)


class AudioStyleEncoder(nn.Module):
    """Projects a precomputed MuQ-MuLan audio/style embedding (the same 512-dim
    contrastive audio-style space DiffRhythm2's teacher itself conditions on)
    into the model's internal conditioning dimension.

    This replaced an earlier version that average-pooled a raw mel crop of the
    backing track with an untrained Conv1D -- that threw away all temporal
    structure and had nothing to do with any learned notion of musical style.
    Using the real MuLan embedding both gives the student a far richer style
    signal and lets the *same* embedding be handed unmodified to the teacher
    during distillation (see docs/experiments/distillation_fix.md).
    """
    def __init__(self, style_dim: int, dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(style_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, style_anchor: torch.Tensor) -> torch.Tensor:
        # Expected input shape: (batch_size, style_dim) -- a single embedding vector per item.
        return self.fc(style_anchor)


class AdaLayerNormZeroFinal(nn.Module):
    """Adaptive layer norm for modulating output feature maps using timestep features."""
    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(cond_dim, dim * 2)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        scale, shift = torch.chunk(self.linear(self.silu(emb)), 2, dim=-1)
        return self.norm(x) * (1 + scale).unsqueeze(1) + shift.unsqueeze(1)


class MicroDiT(nn.Module):
    """A highly optimized, shrunken Diffusion Transformer (DiT) utilizing Llama blocks for music generation."""
    def __init__(
        self,
        config: MusicDiffusionConfig,
        roberta_model: str = "xlm-roberta-base",
        dim: int = 256,
        depth: int = 4,
        heads: int = 4,
        ff_mult: int = 4,
        style_dim: int = 512,
    ):
        super().__init__()
        self.config = config
        self.dim = dim
        self.cond_dim = dim
        self.style_dim = style_dim

        # Core embeddings and adapters
        self.text_encoder = PretrainedRobertaEncoder(model_name=roberta_model, out_dim=dim)
        self.time_embed = TimestepEmbedding(self.cond_dim)
        self.audio_style_encoder = AudioStyleEncoder(style_dim, dim)
        self.style_embed = nn.Sequential(
            nn.Linear(dim, self.cond_dim),
            nn.SiLU(),
            nn.Linear(self.cond_dim, self.cond_dim)
        )
        
        self.input_embed = InputEmbedding(config.n_mels, dim, dim, cond_dim=self.cond_dim)
        
        # Llama decoding blocks
        llama_config = LlamaConfig(
            hidden_size=dim,
            intermediate_size=dim * ff_mult,
            num_attention_heads=heads,
            hidden_act="silu",
            max_position_embeddings=2048
        )
        llama_config._attn_implementation = "sdpa" # FlashAttention speedup
        
        self.transformer_blocks = nn.ModuleList(
            [LlamaDecoderLayer(llama_config, layer_idx=i) for i in range(depth)]
        )
        self.rotary_emb = LlamaRotaryEmbedding(config=llama_config)
        
        # Adaptive residual text fusion layers
        self.text_fusion_linears = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, dim),
                    nn.SiLU()
                ) for _ in range(depth)
            ]
        )
        
        # Zero out initial residual weights
        for layer in self.text_fusion_linears:
            for p in layer.parameters():
                p.data.zero_()
                
        self.norm_out = AdaLayerNormZeroFinal(dim, self.cond_dim)
        self.proj_out = nn.Linear(dim, config.n_mels)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        texts: list[str],
        timestep: torch.Tensor,
        style_prompt: torch.Tensor | None = None
    ) -> torch.Tensor:
        device = x.device
        batch_size, seq_len = x.shape[0], x.shape[1]
        
        # 1. Encode text via RoBERTa (frozen)
        text_embeds, _ = self.text_encoder(texts, device) # (batch_size, text_seq_len, dim)
        pooled_text = text_embeds.mean(dim=1)
        
        # 2. Compute conditional vectors
        t_emb = self.time_embed(timestep) # (batch_size, cond_dim)
        
        if style_prompt is not None:
            style_vector = self.audio_style_encoder(style_prompt)
        else:
            style_vector = pooled_text
            
        s_emb = self.style_embed(style_vector) # (batch_size, cond_dim)
        c = t_emb + s_emb
        
        # Align text embedding length to seq_len for feature merging
        text_embeds_padded = nn.functional.interpolate(
            text_embeds.transpose(1, 2), size=seq_len, mode="linear", align_corners=False
        ).transpose(1, 2)
        
        # 3. Project and embed features
        x = self.input_embed(x, cond, text_embeds_padded, t_emb, s_emb)
        
        # 4. Apply Llama Transformer Blocks
        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0).repeat(batch_size, 1)
        rotary_embed = self.rotary_emb(x, pos_ids)
        
        # Build self-attention mask (noncausal)
        attention_mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=device)
        attention_mask_4d = attention_mask.unsqueeze(1).unsqueeze(2).repeat(1, 1, seq_len, 1)
        attention_mask_inverted = (~attention_mask_4d).float() * torch.finfo(x.dtype).min
        
        for i, block in enumerate(self.transformer_blocks):
            # Feed through llama block
            x, *_ = block(x, attention_mask=attention_mask_inverted, position_embeddings=rotary_embed)
            # Add adaptive text residuals
            x = x + self.text_fusion_linears[i](text_embeds_padded)
            
        # 5. Modulate and project out
        x = self.norm_out(x, c)
        return self.proj_out(x)
