import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModel
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRotaryEmbedding, apply_rotary_pos_emb, repeat_kv
from transformers.models.llama import LlamaConfig

from .text_to_music_diffusion import MusicDiffusionConfig

class PretrainedPhonemeEncoder(nn.Module):
    """Frozen pretrained XPhoneBERT text encoder to extract rich semantic phoneme-level sequence embeddings."""
    def __init__(self, model_name: str = "vinai/xphonebert-base", out_dim: int = 256):
        super().__init__()
        print(f"Loading pretrained XPhoneBERT phoneme encoder: {model_name}...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.roberta = AutoModel.from_pretrained(model_name)
        
        # Freeze all XPhoneBERT parameters
        for param in self.roberta.parameters():
            param.requires_grad = False
            
        self.projection = nn.Sequential(
            nn.Linear(self.roberta.config.hidden_size, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim)
        )

    def train(self, mode: bool = True):
        # The projection remains trainable, but the frozen backbone must stay in
        # eval mode or XPhoneBERT dropout makes the same lyric embedding fluctuate
        # between optimization steps.
        super().train(mode)
        self.roberta.eval()
        return self

    def forward(self, texts: list[str], device) -> tuple[torch.Tensor, torch.Tensor]:
        # Lazily initialize G2P model with appropriate CUDA setting
        if not hasattr(self, "text2phone_model"):
            from text2phonemesequence import Text2PhonemeSequence
            is_cuda = "cuda" in str(device)
            self.text2phone_model = Text2PhonemeSequence(language='vie', is_cuda=is_cuda)

        # Convert texts to phoneme sequences
        phoneme_texts = []
        for text in texts:
            try:
                phonemes = self.text2phone_model.infer_sentence(text)
            except Exception:
                phonemes = text
            phoneme_texts.append(phonemes)

        # Tokenize inputs
        inputs = self.tokenizer(phoneme_texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        
        # Extract embeddings from XPhoneBERT (no gradients computed)
        with torch.no_grad():
            outputs = self.roberta(**inputs)
            # Use sequence output (batch_size, seq_len, hidden_size)
            seq_embeddings = outputs.last_hidden_state
            
        # Project to target dimension
        projected = self.projection(seq_embeddings)
        attention_mask = inputs["attention_mask"].bool() # (batch_size, seq_len)
        return projected, attention_mask



class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding generator supporting 1D and 2D inputs."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        is_1d = timestep.dim() == 1
        if is_1d:
            timestep = timestep.unsqueeze(-1)
            
        device = timestep.device
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = timestep.unsqueeze(-1) * emb.unsqueeze(0).unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        out = self.mlp(emb)
        if is_1d:
            out = out.squeeze(1)
        return out


class InputEmbedding(nn.Module):
    """Mel and Text projection block with sequence concatenation and additive conditioning."""
    def __init__(self, mel_dim: int, text_dim: int, out_dim: int):
        super().__init__()
        # Project mel (x) to out_dim
        self.proj_x = nn.Linear(mel_dim, out_dim)
        # Project text_embed to out_dim if shapes differ
        self.proj_text = nn.Linear(text_dim, out_dim) if text_dim != out_dim else nn.Identity()
        self.proj_final = nn.Linear(out_dim, out_dim)

    def forward(self, x: torch.Tensor, text_embed: torch.Tensor, time_emb: torch.Tensor, style_emb: torch.Tensor) -> torch.Tensor:
        # Project components to the same out_dim space
        x_proj = self.proj_x(x)
        text_proj = self.proj_text(text_embed)
        
        # Concatenate text tokens and mel frames along the sequence dimension (dim=1)
        x_seq = torch.cat([text_proj, x_proj], dim=1)
        
        # Expand style across total sequence length
        seq_len = x_seq.shape[1]
        style_emb_expanded = style_emb.unsqueeze(1).repeat(1, seq_len, 1)
        
        # Sum the representations directly (time_emb is already 2D of shape [B, total_len, out_dim])
        merged = x_seq + style_emb_expanded + time_emb
        return self.proj_final(merged)


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
        roberta_model: str = "vinai/xphonebert-base",
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
        self.text_encoder = PretrainedPhonemeEncoder(model_name=roberta_model, out_dim=dim)
        self.time_embed = TimestepEmbedding(self.cond_dim)
        self.audio_style_encoder = AudioStyleEncoder(style_dim, dim)
        self.style_embed = nn.Sequential(
            nn.Linear(dim, self.cond_dim),
            nn.SiLU(),
            nn.Linear(self.cond_dim, self.cond_dim)
        )
        
        self.input_embed = InputEmbedding(config.n_mels, dim, dim)
        
        # Llama decoding blocks
        llama_config = LlamaConfig(
            hidden_size=dim,
            intermediate_size=dim * ff_mult,
            num_attention_heads=heads,
            hidden_act="silu",
            max_position_embeddings=2048
        )
        # "eager" (not "sdpa") is required so self_attn actually materializes and
        # returns attention weights -- sdpa's fused kernel always returns
        # (attn_output, None) regardless of output_attentions (see
        # transformers.integrations.sdpa_attention.sdpa_attention_forward), which
        # would silently break attention-matrix distillation (see
        # docs/PROJECT_REPORT.md's attention-distillation section).
        llama_config._attn_implementation = "eager"
        
        self.transformer_blocks = nn.ModuleList(
            [LlamaDecoderLayer(llama_config, layer_idx=i) for i in range(depth)]
        )
        self.rotary_emb = LlamaRotaryEmbedding(config=llama_config)
        
        self.norm_out = AdaLayerNormZeroFinal(dim, self.cond_dim)
        self.proj_out = nn.Linear(dim, config.n_mels)

    def forward(
        self,
        x: torch.Tensor,
        texts: list[str],
        timestep: torch.Tensor,
        style_prompt: torch.Tensor | None = None,
        return_attentions: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        device = x.device
        batch_size, seq_len = x.shape[0], x.shape[1]
        
        # 1. Encode text via RoBERTa (frozen)
        text_embeds, text_mask = self.text_encoder(texts, device) # (batch_size, text_seq_len, dim)
        mask = text_mask.unsqueeze(-1).to(text_embeds.dtype)
        pooled_text = (text_embeds * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        
        # 2. Compute conditional vectors
        t_emb_scalar = self.time_embed(timestep) # (batch_size, cond_dim)
        
        # Prepare 2D time sequence: text has -1.0, audio has the scalar timestep
        text_time = torch.full((batch_size, text_embeds.shape[1]), -1.0, device=device, dtype=x.dtype)
        noisy_time = timestep.unsqueeze(-1).repeat(1, seq_len)
        time = torch.cat([text_time, noisy_time], dim=1)
        t_emb_2d = self.time_embed(time) # (batch_size, text_len + seq_len, cond_dim)
        
        if style_prompt is not None:
            style_vector = self.audio_style_encoder(style_prompt)
        else:
            style_vector = pooled_text
            
        s_emb = self.style_embed(style_vector) # (batch_size, cond_dim)
        c = t_emb_scalar + s_emb
        
        # 3. Project and embed features (InputEmbedding concatenates and adds time/style)
        x = self.input_embed(x, text_embeds, t_emb_2d, s_emb)
        
        # 4. Prepare position IDs for concatenated text + mel sequence
        text_len = text_embeds.shape[1]
        text_position_ids = torch.arange(text_len, device=device).unsqueeze(0).repeat(batch_size, 1)
        noisy_position_ids = torch.arange(seq_len, device=device).unsqueeze(0).repeat(batch_size, 1)
        pos_ids = torch.cat([text_position_ids, noisy_position_ids], dim=1)
        
        rotary_embed = self.rotary_emb(x, pos_ids)
        
        # 5. Build self-attention mask (noncausal bidirectional attention)
        key_valid = torch.cat([text_mask, torch.ones((batch_size, seq_len), dtype=torch.bool, device=device)], dim=1)
        total_len = key_valid.shape[1]
        attention_mask_4d = key_valid.unsqueeze(1).unsqueeze(2).repeat(1, 1, total_len, 1)
        attention_mask_inverted = (~attention_mask_4d).float() * torch.finfo(x.dtype).min
        
        # LlamaDecoderLayer.forward (this transformers version) computes attn_weights
        # inside self.self_attn but discards them (`hidden_states, _ = self.self_attn(...)`)
        # before returning. Reading them off self_attn's own return value via a plain
        # forward hook is *also* unreliable: it depends on the resolved attention
        # implementation (config._attn_implementation) actually being "eager" at
        # call time, which was observed to silently NOT hold on a real Kaggle GPU run
        # despite setting it on the LlamaConfig before constructing the layers (see
        # docs/PROJECT_REPORT.md's attention-distillation section -- local CPU test
        # passed, real CUDA run returned None for every layer, and no GPU is
        # available locally to root-cause the discrepancy further). Recomputing the
        # attention weights directly from self_attn's own q_proj/k_proj -- the exact
        # same computation eager_attention_forward does -- sidesteps that dispatch
        # entirely and is correct regardless of which implementation the module
        # itself ends up using internally for its own output.
        captured_attentions: list[torch.Tensor] = []
        hook_handles = []
        if return_attentions:
            def _capture(module, _args, kwargs):
                hidden_states = kwargs["hidden_states"]
                position_embeddings = kwargs["position_embeddings"]
                attention_mask = kwargs.get("attention_mask")
                b, seq_len_local = hidden_states.shape[0], hidden_states.shape[1]
                head_dim = module.head_dim
                q = module.q_proj(hidden_states).view(b, seq_len_local, -1, head_dim).transpose(1, 2)
                k = module.k_proj(hidden_states).view(b, seq_len_local, -1, head_dim).transpose(1, 2)
                cos, sin = position_embeddings
                q, k = apply_rotary_pos_emb(q, k, cos, sin)
                if getattr(module, "num_key_value_groups", 1) > 1:
                    k = repeat_kv(k, module.num_key_value_groups)
                # head_dim**-0.5 is the standard attention scaling factor computed
                # directly rather than read off module.scaling -- that attribute's
                # name/presence varies across transformers versions (confirmed: it
                # exists locally but the version actually installed on the Kaggle
                # kernel from DiffRhythm2's requirements.txt raised AttributeError
                # for it), so recomputing the well-defined constant here is more
                # robust than depending on either version's internal naming.
                scaling = head_dim ** -0.5
                scores = (q @ k.transpose(-1, -2)) * scaling
                if attention_mask is not None:
                    scores = scores + attention_mask
                captured_attentions.append(torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype))

            hook_handles = [
                block.self_attn.register_forward_pre_hook(_capture, with_kwargs=True)
                for block in self.transformer_blocks
            ]

        try:
            for i, block in enumerate(self.transformer_blocks):
                # Feed through llama block on the unified sequence
                res = block(x, attention_mask=attention_mask_inverted, position_embeddings=rotary_embed)
                x = res[0] if isinstance(res, tuple) else res
        finally:
            for handle in hook_handles:
                handle.remove()

        # 6. Extract the audio mel portion of the sequence
        x = x[:, text_len:]

        # 7. Modulate and project out
        x = self.norm_out(x, c)
        out = self.proj_out(x)
        if return_attentions:
            return out, captured_attentions, text_len
        return out
