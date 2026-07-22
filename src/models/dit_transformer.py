import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModel
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaMLP, LlamaRMSNorm, LlamaRotaryEmbedding
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
            # 'vie' is not a real CharsiuG2P language tag (confirmed: the
            # package's own default is 'vie-c', and its internal dict only
            # lists vie-c/vie-n/vie-s -- three Vietnamese dialect variants,
            # no bare 'vie'). With 'vie', the wget for vie.tsv 404s, phone_dict
            # stays empty, and every word falls through to the raw '<vie>: '
            # prompt -- a tag the G2P model never saw during its own training.
            # Every phoneme sequence fed to XPhoneBERT has been affected.
            self.text2phone_model = Text2PhonemeSequence(language='vie-c', is_cuda=is_cuda)

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
    """Mel projection block with additive timestep/style conditioning.

    Lyric conditioning is no longer folded in here: the previous scheme
    concatenated text and mel tokens into one self-attention sequence
    ("prepend"-style). SongGen (arXiv:2502.13128) found cross-attention lyric
    conditioning clearly beats that (FAD 1.73 vs 3.56, PER 43.34 vs 56.21), so
    lyric conditioning now happens via a dedicated cross-attention sublayer in
    each transformer block instead (see CrossAttentionDecoderLayer below) --
    this embedding only ever sees the mel sequence.
    """
    def __init__(self, mel_dim: int, out_dim: int):
        super().__init__()
        self.proj_x = nn.Linear(mel_dim, out_dim)
        self.proj_final = nn.Linear(out_dim, out_dim)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor, style_emb: torch.Tensor) -> torch.Tensor:
        x_proj = self.proj_x(x)
        seq_len = x_proj.shape[1]
        style_emb_expanded = style_emb.unsqueeze(1).repeat(1, seq_len, 1)
        time_emb_expanded = time_emb.unsqueeze(1).repeat(1, seq_len, 1)
        merged = x_proj + style_emb_expanded + time_emb_expanded
        return self.proj_final(merged)


class CrossAttentionDecoderLayer(nn.Module):
    """A Llama-style block over the mel sequence only, with an added
    cross-attention sublayer attending to lyric/text token embeddings.

    Replaces the previous "prepend" scheme (text and mel tokens concatenated
    into one shared self-attention sequence, see InputEmbedding's old
    docstring) with the architecture SongGen (arXiv:2502.13128) found works
    better: self-attention stays within the mel sequence (so rotary positions
    and the attention mask are just plain mel-frame positions, no more text
    offset/padding bookkeeping), and a dedicated cross-attention sublayer lets
    mel queries pull in lyric content from text keys/values. Reuses Llama's
    own self-attention/MLP/RMSNorm building blocks; cross-attention itself has
    no Llama equivalent (LlamaAttention is hardcoded to self-attention), so
    it's a standard nn.MultiheadAttention sublayer instead.
    """
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.cross_attn_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.cross_attn = nn.MultiheadAttention(
            config.hidden_size, config.num_attention_heads, batch_first=True,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        text_embeds: torch.Tensor,
        text_key_padding_mask: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        # 1. Self-attention over mel tokens only (no padding among mel frames,
        # so no attention mask is needed here -- fully bidirectional).
        residual = hidden_states
        normed = self.input_layernorm(hidden_states)
        # LlamaAttention.forward()'s return tuple length varies across
        # transformers versions (2-tuple vs 3-tuple with past_key_value) --
        # train-distill's kernel installs DiffRhythm2's requirements.txt, which
        # pins a different transformers version than train-self's kernel, so a
        # rigid `a, b = ...` unpack breaks on one of the two. Only the first
        # element (the attention output) is ever needed here.
        self_attn_result = self.self_attn(hidden_states=normed, position_embeddings=position_embeddings)
        attn_out = self_attn_result[0] if isinstance(self_attn_result, tuple) else self_attn_result
        hidden_states = residual + attn_out

        # 2. Cross-attention: mel queries attend to lyric/text keys+values.
        # nn.MultiheadAttention's key_padding_mask uses True == ignore, the
        # opposite convention from text_key_padding_mask (True == valid token).
        residual = hidden_states
        normed = self.cross_attn_layernorm(hidden_states)
        cross_out, _ = self.cross_attn(
            query=normed, key=text_embeds, value=text_embeds,
            key_padding_mask=~text_key_padding_mask, need_weights=False,
        )
        hidden_states = residual + cross_out

        # 3. Feed-forward
        residual = hidden_states
        normed = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(normed)
        return hidden_states


class AudioStyleEncoder(nn.Module):
    """Projects a precomputed MuQ-MuLan audio/style embedding (the same 512-dim
    contrastive audio-style space DiffRhythm2's teacher itself conditions on)
    into the model's internal conditioning dimension.

    This replaced an earlier version that average-pooled a raw mel crop of the
    backing track with an untrained Conv1D -- that threw away all temporal
    structure and had nothing to do with any learned notion of musical style.
    Using the real MuLan embedding both gives the student a far richer style
    signal and lets the *same* embedding be handed unmodified to the teacher
    during distillation (see docs/project_history.md).
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
        repa_dim: int = 1024,
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

        self.input_embed = InputEmbedding(config.n_mels, dim)
        # Projects an intermediate transformer hidden state up to a frozen
        # self-supervised audio encoder's feature dimension (REPA-style
        # representation-alignment auxiliary loss, see docs/project_history.md --
        # mirrors DiffRhythm2's own "Stochastic Block REPA"). Unconditionally
        # constructed (matches this file's existing convention for
        # AudioStyleEncoder/PretrainedPhonemeEncoder); negligible cost, and only
        # ever used when a caller actually passes repa_layer_idx to forward().
        self.repa_head = nn.Sequential(
            nn.Linear(dim, repa_dim),
            nn.SiLU(),
            nn.Linear(repa_dim, repa_dim),
        )
        
        # Llama decoding blocks
        llama_config = LlamaConfig(
            hidden_size=dim,
            intermediate_size=dim * ff_mult,
            num_attention_heads=heads,
            hidden_act="silu",
            max_position_embeddings=2048
        )
        # A bare LlamaConfig() leaves _attn_implementation as None (it's normally
        # set by PreTrainedModel.__init__/from_pretrained, which we bypass here) --
        # modeling_llama.py then does ALL_ATTENTION_FUNCTIONS[None] unconditionally
        # and crashes with KeyError. "sdpa" is the fast default; "eager" was only
        # ever needed by the now-removed attention-distillation capture hook.
        llama_config._attn_implementation = "sdpa"
        self.transformer_blocks = nn.ModuleList(
            [CrossAttentionDecoderLayer(llama_config, layer_idx=i) for i in range(depth)]
        )
        self.rotary_emb = LlamaRotaryEmbedding(config=llama_config)
        
        self.norm_out = AdaLayerNormZeroFinal(dim, self.cond_dim)
        self.proj_out = nn.Linear(dim, config.n_mels)
        # Auxiliary vocal-only prediction head ("Mixed Pro" in SongGen,
        # arXiv:2502.13128): the model's primary target is now the full song
        # (vocal + accompaniment, see reconstruct_full_mix in
        # text_to_music_diffusion.py), and joint mixed-audio targets let a
        # model neglect the harder, sparser vocal signal in favor of the
        # louder, more predictable accompaniment. This head shares the final
        # hidden state with proj_out but predicts vocal-only velocity as an
        # auxiliary training signal only -- it is never used at inference
        # unless a caller explicitly asks for it.
        self.vocal_proj_out = nn.Linear(dim, config.n_mels)

    def forward(
        self,
        x: torch.Tensor,
        texts: list[str],
        timestep: torch.Tensor,
        style_prompt: torch.Tensor | None = None,
        repa_layer_idx: int | None = None,
        return_vocal_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        device = x.device
        batch_size, seq_len = x.shape[0], x.shape[1]

        # 1. Encode text via RoBERTa (frozen) -- kept as cross-attention keys/values
        text_embeds, text_mask = self.text_encoder(texts, device) # (batch_size, text_seq_len, dim)
        mask = text_mask.unsqueeze(-1).to(text_embeds.dtype)
        pooled_text = (text_embeds * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

        # 2. Compute conditional vectors
        t_emb_scalar = self.time_embed(timestep) # (batch_size, cond_dim)

        if style_prompt is not None:
            style_vector = self.audio_style_encoder(style_prompt)
        else:
            style_vector = pooled_text

        s_emb = self.style_embed(style_vector) # (batch_size, cond_dim)
        c = t_emb_scalar + s_emb

        # 3. Project and embed the mel sequence only (lyric conditioning now
        # happens via cross-attention inside each block, not concatenation).
        x = self.input_embed(x, t_emb_scalar, s_emb)

        # 4. Rotary position IDs over the mel sequence only
        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0).repeat(batch_size, 1)
        rotary_embed = self.rotary_emb(x, pos_ids)

        repa_hidden_raw = None
        for i, block in enumerate(self.transformer_blocks):
            x = block(
                x, text_embeds=text_embeds, text_key_padding_mask=text_mask,
                position_embeddings=rotary_embed,
            )
            if repa_layer_idx is not None and i == repa_layer_idx:
                # Hidden state right after this block, before the final
                # AdaLN/proj_out -- this is the "representation" a REPA-style
                # loss aligns with a frozen SSL encoder's features (see
                # docs/project_history.md), analogous to DiffRhythm2's own
                # "Stochastic Block REPA" against its DiT's hidden states.
                repa_hidden_raw = x

        # 5. Modulate and project out
        x = self.norm_out(x, c)
        out = self.proj_out(x)

        result: tuple = (out,)
        if repa_layer_idx is not None:
            repa_projected = self.repa_head(repa_hidden_raw) if repa_hidden_raw is not None else None
            result = result + (repa_projected,)
        if return_vocal_aux:
            # Shares the same AdaLN-modulated hidden state as the main head --
            # only the final linear projection differs (see __init__).
            result = result + (self.vocal_proj_out(x),)
        return result if len(result) > 1 else out
