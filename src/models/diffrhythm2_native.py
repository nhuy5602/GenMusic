# Portions of this file (SinusPositionEmbedding, TextEmbedding, InputEmbedding,
# AdaLayerNormZero_Final, TimestepEmbedding, LlamaAttention, LlamaSdpaAttention,
# LLAMA_ATTENTION_CLASSES, LlamaNARDecoderLayer) are vendored, near-verbatim,
# from ASLP-lab/DiffRhythm2 (https://github.com/ASLP-lab/DiffRhythm2),
# Copyright 2025 ASLP Lab and Xiaomi Inc., licensed under the Apache License,
# Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0).
"""A small student model that reuses DiffRhythm2's *actual* DiT architecture
(concatenated [text_tokens; noisy_mel] self-attention sequence, raw learned
token embeddings, no cross-attention) instead of this project's own
hand-rolled `MicroDiT` (dit_transformer.py, which uses cross-attention lyric
conditioning -- a deliberate SongGen-inspired departure from DiffRhythm2's own
scheme). This exists to test a specific hypothesis: that our own
re-implementation of the teacher's call contract (see
docs/project_history.md and the `950035a` block-wise-mask commit)
has drifted from a documented, simpler design into an undocumented one with a
real train/inference task mismatch (later "noisy" positions get clean,
un-noised context from earlier positions of the SAME training crop -- context
the student's own forward pass never has at generation time). Using the real
architecture end-to-end (for both the student here and, eventually, the
teacher-query path) removes an entire class of "did we replicate the protocol
correctly" bugs, since there is only one real implementation to keep in sync
with, not a parallel hand-written approximation.

Deliberately kept at the *same small scale* as MicroDiT (dim=256/384, depth=4,
few heads) -- this is not a weight-transfer/pruning experiment (see
docs/project_history.md §5.3 item 1 for why that is a separate, much larger
undertaking due to head_dim/RoPE mismatch), just the real architecture trained
from scratch on the same 250-song data budget.

Text conditioning: the real DiT's `TextEmbedding` is a raw `nn.Embedding`
trained from scratch (no pretrained language model at all -- DiffRhythm2's own
`CNENTokenizer` is a fixed-vocabulary Chinese/English G2P frontend). To keep a
Vietnamese-aware vocabulary without reintroducing MicroDiT's frozen-XPhoneBERT
cross-attention machinery, this reuses the existing G2P step
(`text2phonemesequence`, see PretrainedPhonemeEncoder in dit_transformer.py)
to turn lyrics into a phoneme string, then XPhoneBERT's *tokenizer only* (not
its pretrained transformer weights) to turn that string into integer ids --
the resulting embedding table is trained from scratch, exactly matching the
real DiT's own recipe, just with phoneme-level Vietnamese ids instead of
CNENTokenizer's Chinese/English ones.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoTokenizer
from transformers.models.llama.modeling_llama import (
    Cache,
    LlamaConfig,
    LlamaDecoderLayer,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
)

from .text_to_music_diffusion import MusicDiffusionConfig

DIFFRHYTHM2_COND_DIM = 512


# --------------------------------------------------------------------------
# Vendored from diffrhythm2/backbones/dit.py
# --------------------------------------------------------------------------

class TextEmbedding(nn.Module):
    def __init__(self, text_num_embeds: int, text_dim: int):
        super().__init__()
        self.text_embed = nn.Embedding(text_num_embeds + 1, text_dim)  # 0 = filler token

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        return self.text_embed(text)


class InputEmbedding(nn.Module):
    def __init__(self, cond_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(cond_dim, cond_dim)
        self.proj_2 = nn.Linear(cond_dim, out_dim)

    def forward(self, x: torch.Tensor, style_emb: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        style_emb = style_emb.unsqueeze(1).repeat(1, x.shape[1], 1)
        x_orig = x
        x = x + style_emb + time_emb
        x = self.proj(x) + x_orig
        return self.proj_2(x)


class AdaLayerNormZero_Final(nn.Module):
    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(cond_dim, dim * 2)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        emb = self.linear(self.silu(emb))
        scale, shift = torch.chunk(emb, 2, dim=-1)
        return self.norm(x) * (1 + scale) + shift


class SinusPositionEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor, scale: float = 1000) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(-1) * emb.unsqueeze(0).unsqueeze(0)
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class TimestepEmbedding(nn.Module):
    def __init__(self, dim: int, freq_embed_dim: int = 256):
        super().__init__()
        self.time_embed = SinusPositionEmbedding(freq_embed_dim)
        self.time_mlp = nn.Sequential(nn.Linear(freq_embed_dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        time_hidden = self.time_embed(timestep)
        time_hidden = time_hidden.to(timestep.dtype)
        return self.time_mlp(time_hidden)


# --------------------------------------------------------------------------
# Vendored from diffrhythm2/backbones/llama_attention.py (sdpa/eager only --
# flash-attention/flex-attention variants dropped, not needed on Kaggle T4/CPU)
# --------------------------------------------------------------------------

class LlamaAttention(nn.Module):
    """Multi-headed attention, adapted from DiffRhythm2's own LlamaAttention
    (itself adapted from HF transformers) -- adds a learned q/k RMSNorm on top
    of standard Llama self-attention."""

    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        self.rotary_emb = LlamaRotaryEmbedding(config=self.config)
        self.q_norm = LlamaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = LlamaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def _qkv(self, hidden_states, position_embeddings, past_key_value):
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states).view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if past_key_value is not None:
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, {"sin": sin, "cos": cos}
            )
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        return query_states, key_states, value_states, bsz, q_len

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_value: Optional["Cache"] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        query_states, key_states, value_states, bsz, q_len = self._qkv(hidden_states, position_embeddings, past_key_value)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            if attention_mask.dtype != torch.bool:
                attn_weights = attn_weights + causal_mask
            else:
                attn_weights = torch.masked_fill(attn_weights, ~causal_mask, float("-inf"))
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, (attn_weights if output_attentions else None), past_key_value


class LlamaSdpaAttention(LlamaAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_value: Optional["Cache"] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        if output_attentions:
            return super().forward(
                hidden_states=hidden_states, attention_mask=attention_mask,
                position_embeddings=position_embeddings, past_key_value=past_key_value,
                output_attentions=True, use_cache=use_cache,
            )
        query_states, key_states, value_states, bsz, q_len = self._qkv(hidden_states, position_embeddings, past_key_value)
        causal_mask = attention_mask
        if causal_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states, key_states, value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=False,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, None, past_key_value


LLAMA_ATTENTION_CLASSES = {"eager": LlamaAttention, "sdpa": LlamaSdpaAttention}


# --------------------------------------------------------------------------
# Vendored from diffrhythm2/backbones/llama_nar.py
# --------------------------------------------------------------------------

class LlamaNARDecoderLayer(LlamaDecoderLayer):
    """Non-autoregressive Llama decoder layer: identical to a standard Llama
    block, just with `self_attn` swapped for a non-causal variant (`is_causal`
    is never set True in the attention classes above) since CFM denoises a
    whole sequence at once, not token-by-token."""

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = LLAMA_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states, attention_mask=attention_mask,
            position_embeddings=position_embeddings, past_key_value=past_key_value,
            output_attentions=output_attentions, use_cache=use_cache,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


# --------------------------------------------------------------------------
# Our own wrapper: real DiT architecture + Vietnamese phoneme vocabulary +
# this project's training-loop contract (same forward() signature as MicroDiT
# so it drops into the existing cfm_loss()/DiffusionTrainer/train_model()
# infrastructure unchanged).
# --------------------------------------------------------------------------

class NativeDiTStudent(nn.Module):
    """DiffRhythm2's real DiT architecture (concatenated self-attention over
    [text_tokens; noisy_mel], raw learned token embeddings, no cross-attention)
    at student scale, trained from scratch on Vietnamese phoneme tokens.

    Deliberately uses the SIMPLE, fully-visible (non-block) attention contract
    documented in docs/project_history.md -- text and audio
    positions all attend to each other freely (modulo text padding), with NO
    clean/noisy block split. This avoids the same-crop clean-context leak
    found in `_teacher_velocity`/`_build_block_attn_mask`
    (src/training/distill_training.py): every position here sees the same
    noise level `t`, exactly matching what `loss_gt`'s single global timestep
    assumes.
    """

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
        cond_dim = DIFFRHYTHM2_COND_DIM

        self.tokenizer = AutoTokenizer.from_pretrained(roberta_model)
        self.text_embed = TextEmbedding(self.tokenizer.vocab_size, cond_dim)
        self.time_embed = TimestepEmbedding(cond_dim)
        self.latent_embed = nn.Sequential(nn.Linear(config.n_mels, cond_dim), nn.Linear(cond_dim, cond_dim))
        self.input_embed = InputEmbedding(cond_dim, dim)
        self.style_proj = nn.Linear(style_dim, cond_dim) if style_dim != cond_dim else nn.Identity()

        llama_config = LlamaConfig(
            hidden_size=dim, num_attention_heads=heads, intermediate_size=dim * ff_mult,
            hidden_act="silu", max_position_embeddings=4096,
        )
        llama_config._attn_implementation = "sdpa"
        self.transformer_blocks = nn.ModuleList(
            [LlamaNARDecoderLayer(llama_config, layer_idx=i) for i in range(depth)]
        )
        self.rotary_embed = LlamaRotaryEmbedding(config=llama_config)

        self.norm_out = AdaLayerNormZero_Final(dim, cond_dim)
        self.proj_out = nn.Linear(dim, config.n_mels)
        self.vocal_proj_out = nn.Linear(dim, config.n_mels)

        self._text2phone_model = None

    def _phonemize(self, texts: list[str], device) -> tuple[torch.Tensor, torch.Tensor]:
        if self._text2phone_model is None:
            from text2phonemesequence import Text2PhonemeSequence

            self._text2phone_model = Text2PhonemeSequence(language="vie-c", is_cuda="cuda" in str(device))
        phoneme_texts = []
        for text in texts:
            try:
                phonemes = self._text2phone_model.infer_sentence(text)
            except Exception:
                phonemes = text
            phoneme_texts.append(phonemes)
        encoded = self.tokenizer(phoneme_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
        return encoded["input_ids"].to(device), encoded["attention_mask"].bool().to(device)

    def forward(
        self,
        x: torch.Tensor,
        texts: list[str],
        timestep: torch.Tensor,
        style_prompt: torch.Tensor | None = None,
        repa_layer_idx: int | None = None,
        return_vocal_aux: bool = False,
    ):
        device = x.device
        batch_size, audio_len = x.shape[0], x.shape[1]

        token_ids, token_valid = self._phonemize(texts, device)
        text_len = token_ids.shape[1]

        text_emb = self.text_embed(token_ids)  # (B, text_len, cond_dim)
        audio_emb = self.latent_embed(x)  # (B, audio_len, cond_dim)
        combined = torch.cat([text_emb, audio_emb], dim=1)  # (B, text_len+audio_len, cond_dim)

        text_time = torch.full((batch_size, text_len), -1.0, device=device, dtype=x.dtype)
        audio_time = timestep[:, None].repeat(1, audio_len).to(x.dtype)
        time = torch.cat([text_time, audio_time], dim=1)

        # Position ids restart at 0 for the audio segment (matches DiffRhythm2's
        # own first-inference-block behavior, see docs/project_history.md 
        # distillation_fix.md's "known residual limitations" -- both segments
        # get their own independent 0..N-1 range, not one continuing range).
        text_position_ids = torch.arange(text_len, device=device).unsqueeze(0).repeat(batch_size, 1)
        audio_position_ids = torch.arange(audio_len, device=device).unsqueeze(0).repeat(batch_size, 1)
        position_ids = torch.cat([text_position_ids, audio_position_ids], dim=1)

        total_len = text_len + audio_len
        key_valid = torch.cat(
            [token_valid, torch.ones(batch_size, audio_len, dtype=torch.bool, device=device)], dim=1
        )
        attn_mask = key_valid[:, None, None, :].repeat(1, 1, total_len, 1)  # (B,1,Q,KV), True=attend

        style_vector = self.style_proj(style_prompt) if style_prompt is not None else combined.new_zeros(batch_size, DIFFRHYTHM2_COND_DIM)

        t_emb = self.time_embed(time)  # (B, total_len, cond_dim)
        hidden = self.input_embed(combined, style_vector, t_emb)  # (B, total_len, dim)

        position_embeddings = self.rotary_embed(hidden, position_ids)

        repa_hidden = None
        for i, block in enumerate(self.transformer_blocks):
            hidden = block(hidden, attention_mask=attn_mask, position_embeddings=position_embeddings)
            if repa_layer_idx is not None and i == repa_layer_idx:
                repa_hidden = hidden

        # adaLN conditioning uses only the timestep, matching the real DiT
        # (see dit.py: `c = t`, style is injected once at input_embed only).
        modulated = self.norm_out(hidden, t_emb)
        out = self.proj_out(modulated)
        predicted_velocity = out[:, text_len:, :]  # drop the text positions' output

        result: tuple = (predicted_velocity,)
        if repa_layer_idx is not None:
            result = result + (repa_hidden[:, text_len:, :] if repa_hidden is not None else None,)
        if return_vocal_aux:
            vocal_aux = self.vocal_proj_out(modulated)[:, text_len:, :]
            result = result + (vocal_aux,)
        return result if len(result) > 1 else predicted_velocity
