import torch
import os

from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModel
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaMLP, LlamaRMSNorm, LlamaRotaryEmbedding
from transformers.models.llama import LlamaConfig

from .text_to_music_diffusion import MusicDiffusionConfig
from .native_text import (
    NativeAudioTextRecognizer,
    NativeProsodyConditioner,
    NativeVietnameseTextEncoder,
    NativeVocalMelPrior,
)


def align_text_to_frames(
    text_embed: torch.Tensor,
    text_mask: torch.Tensor,
    text_present: torch.Tensor,
    frame_count: int,
) -> torch.Tensor:
    """Monotonically expand ordered lyric tokens onto the audio timeline."""
    aligned_text = []
    for batch_index in range(text_embed.shape[0]):
        valid = text_embed[batch_index][text_mask[batch_index]]
        if valid.shape[0] == 0 or not bool(text_present[batch_index]):
            aligned = torch.zeros(
                (frame_count, text_embed.shape[-1]),
                dtype=text_embed.dtype,
                device=text_embed.device,
            )
        else:
            aligned = F.interpolate(
                valid.transpose(0, 1).unsqueeze(0),
                size=frame_count,
                mode="linear",
                align_corners=False,
            ).squeeze(0).transpose(0, 1)
        aligned_text.append(aligned)
    return torch.stack(aligned_text)

class PretrainedPhonemeEncoder(nn.Module):
    """Frozen pretrained XPhoneBERT text encoder to extract rich semantic phoneme-level sequence embeddings."""
    is_native = False
    encoder_type = "pretrained_xphonebert"
    def __init__(self, model_name: str = "vinai/xphonebert-base", out_dim: int = 256):
        super().__init__()
        resolved_model_name = os.getenv("GENMUSIC_XPHONEBERT_PATH") or model_name
        print(f"Loading pretrained XPhoneBERT phoneme encoder: {resolved_model_name}...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(resolved_model_name)
        self.roberta = AutoModel.from_pretrained(resolved_model_name)
        
        # Freeze all XPhoneBERT parameters
        for param in self.roberta.parameters():
            param.requires_grad = False
            
        self.projection = nn.Sequential(
            nn.Linear(self.roberta.config.hidden_size, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim)
        )
        # Neural Vietnamese G2P is much more expensive than the frozen
        # XPhoneBERT forward pass when the same four-second lyric crop is seen
        # over many epochs. Keep the phoneme string on CPU and reuse it; this
        # does not cache trainable tensors or detach the projection path.
        self._phoneme_cache: dict[str, str] = {}

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
            # Charsiu publishes three explicit Vietnamese frontends. ``vie``
            # is not one of them and silently falls back to an unknown language
            # tag; ``vie-c`` is the package default and the best general-purpose
            # choice when the dataset mixes regional singers.
            self.text2phone_model = Text2PhonemeSequence(
                pretrained_g2p_model=(
                    os.getenv("GENMUSIC_CHARSIU_G2P_PATH")
                    or "charsiu/g2p_multilingual_byT5_small_100"
                ),
                tokenizer=os.getenv("GENMUSIC_BYT5_PATH") or "google/byt5-small",
                language="vie-c",
                is_cuda=is_cuda,
            )

        # Convert texts to phoneme sequences
        phoneme_texts = []
        for text in texts:
            cache_key = str(text)
            phonemes = self._phoneme_cache.get(cache_key)
            if phonemes is None:
                try:
                    phonemes = self.text2phone_model.infer_sentence(cache_key)
                except Exception:
                    # Preserve a deterministic conditioning path if one rare
                    # token cannot be phonemized; XPhoneBERT still receives the
                    # original Vietnamese text instead of dropping the lyric.
                    phonemes = cache_key
                self._phoneme_cache[cache_key] = phonemes
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
    """Mel projection with additive conditions and monotonic lyric alignment.

    Transformer blocks still use dedicated lyric cross-attention. The aligned
    input is a lightweight positional prior that prevents content conditioning
    from collapsing while keeping the mel sequence separate from text tokens.
    """
    def __init__(self, mel_dim: int, out_dim: int, text_dim: int | None = None):
        super().__init__()
        self.proj_x = nn.Linear(mel_dim, out_dim)
        self.proj_text = (
            nn.Linear(text_dim, out_dim)
            if text_dim is not None and text_dim != out_dim
            else nn.Identity()
        )
        self.proj_final = nn.Linear(out_dim, out_dim)
        # The previous token-concatenation-only path let the denoiser ignore
        # lyrics (<1% response when the prompt changed).  A bounded learnable
        # gate injects the ordered phoneme sequence directly into every audio
        # frame.  Its non-zero floor is intentional: validation showed that an
        # unconstrained gate was driven toward zero even as acoustic loss fell.
        # Keeping the same parameter name also preserves checkpoint upgrades.
        self.text_frame_gate = nn.Parameter(torch.tensor(0.25))

    def text_frame_strength(self) -> torch.Tensor:
        """Return a stable 0.20..0.50 strength for frame-aligned lyric input."""
        return 0.20 + 0.30 * torch.sigmoid(self.text_frame_gate)

    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor,
        style_emb: torch.Tensor,
        text_embed: torch.Tensor | None = None,
        text_mask: torch.Tensor | None = None,
        text_present: torch.Tensor | None = None,
        aligned_text: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x_proj = self.proj_x(x)

        if text_embed is not None and text_mask is not None:
            text_embed = self.proj_text(text_embed)
            present = (
                text_present
                if text_present is not None
                else text_mask.any(dim=1)
            )
            aligned = aligned_text
            if aligned is None:
                aligned = align_text_to_frames(
                    text_embed,
                    text_mask,
                    present,
                    x_proj.shape[1],
                )
            x_proj = x_proj + self.text_frame_strength() * aligned

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
        text_encoder_type: str = "native_utf8",
        dim: int = 256,
        depth: int = 4,
        heads: int = 4,
        ff_mult: int = 4,
        style_dim: int = 512,
        repa_dim: int = 1024,
        generation_target: str = "full_mix",
    ):
        super().__init__()
        self.config = config
        self.dim = dim
        self.cond_dim = dim
        self.style_dim = style_dim
        self.generation_target = str(generation_target)
        if self.generation_target not in {"full_mix", "joint_stems"}:
            raise ValueError(
                "generation_target must be 'full_mix' or 'joint_stems', "
                f"got {self.generation_target!r}"
            )

        # Core embeddings and adapters
        self.text_encoder_type = str(text_encoder_type)
        if self.text_encoder_type == "native_utf8":
            self.text_encoder = NativeVietnameseTextEncoder(out_dim=dim)
            # This recognizer receives audio only. Its CTC loss prevents the
            # full-mix denoiser from copying text embeddings without rendering
            # the corresponding Vietnamese content into the vocal component.
            self.audio_text_recognizer = NativeAudioTextRecognizer(config.n_mels, dim)
            self.native_prosody = NativeProsodyConditioner(dim)
            self.native_vocal_prior = NativeVocalMelPrior(dim, config.n_mels)
            # A bounded non-zero residual makes the direct grapheme-to-vocal
            # path effective immediately while leaving the DiT free to model
            # singer, melody and accompaniment variation around it.
            self.native_vocal_prior_gate = nn.Parameter(torch.tensor(-1.0))
        elif self.text_encoder_type == "pretrained_xphonebert":
            self.text_encoder = PretrainedPhonemeEncoder(model_name=roberta_model, out_dim=dim)
            self.audio_text_recognizer = None
            self.native_prosody = None
            self.native_vocal_prior = None
            self.register_parameter("native_vocal_prior_gate", None)
        else:
            raise ValueError(
                "text_encoder_type must be 'native_utf8' or 'pretrained_xphonebert', "
                f"got {self.text_encoder_type!r}"
            )
        self.time_embed = TimestepEmbedding(self.cond_dim)
        self.audio_style_encoder = AudioStyleEncoder(style_dim, dim)
        self.style_embed = nn.Sequential(
            nn.Linear(dim, self.cond_dim),
            nn.SiLU(),
            nn.Linear(self.cond_dim, self.cond_dim)
        )

        model_mels = config.n_mels * (2 if self.joint_stem_generation else 1)
        self.input_embed = InputEmbedding(model_mels, dim)
        # Projects an intermediate transformer hidden state up to a frozen
        # self-supervised audio encoder's feature dimension (REPA-style
        # representation-alignment auxiliary loss, see docs/PROJECT_REPORT.md --
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
        self.proj_out = nn.Linear(dim, model_mels)
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

    @property
    def native_generation(self) -> bool:
        return self.text_encoder_type == "native_utf8"

    @property
    def joint_stem_generation(self) -> bool:
        """Whether one denoising state jointly carries backing and vocal mels."""
        return self.generation_target == "joint_stems"

    def audio_text_logits(self, vocal_mel: torch.Tensor) -> torch.Tensor:
        """Recognize lyrics from audio only for native CTC supervision."""
        if self.audio_text_recognizer is None:
            raise RuntimeError("Audio-text CTC is only available with native_utf8 text encoding")
        return self.audio_text_recognizer(vocal_mel)

    def native_vocal_prior_strength(self) -> torch.Tensor:
        """Return a stable 0.10..1.00 contribution to vocal flow velocity."""
        if self.native_vocal_prior_gate is None:
            return torch.zeros((), device=next(self.parameters()).device)
        return 0.10 + 0.90 * torch.sigmoid(self.native_vocal_prior_gate)

    def forward(
        self,
        x: torch.Tensor,
        texts: list[str],
        timestep: torch.Tensor,
        style_prompt: torch.Tensor | None = None,
        repa_layer_idx: int | None = None,
        return_vocal_aux: bool = False,
        return_native_vocal_prior: bool = False,
        return_native_prosody: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        device = x.device
        batch_size, seq_len = x.shape[0], x.shape[1]

        # 1. Encode text into ordered lyric tokens. Native checkpoints learn
        # this frontend jointly; legacy checkpoints use frozen XPhoneBERT.
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

        # 3. Keep text and mel as separate cross-attention sequences while a
        # monotonic frame-aligned prior makes lyric order explicit at input.
        text_present = torch.tensor(
            [bool(str(text).strip()) for text in texts],
            dtype=torch.bool,
            device=device,
        )
        native_vocal_prior = None
        native_prosody = None
        if self.native_vocal_prior is not None:
            native_prosody = self.native_prosody(
                text_embeds, text_mask, text_present, seq_len
            )
            aligned_native_text = native_prosody["aligned_text"]
            native_vocal_prior = self.native_vocal_prior(
                aligned_native_text,
                text_present,
            )
        else:
            aligned_native_text = None
        x = self.input_embed(
            x,
            t_emb_scalar,
            s_emb,
            text_embed=text_embeds,
            text_mask=text_mask,
            text_present=text_present,
            aligned_text=aligned_native_text,
        )

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
                # docs/PROJECT_REPORT.md), analogous to DiffRhythm2's own
                # "Stochastic Block REPA" against its DiT's hidden states.
                repa_hidden_raw = x

        # 5. Modulate and project out
        x = self.norm_out(x, c)
        out = self.proj_out(x)
        if self.joint_stem_generation:
            # The shared projection is excellent at the dense accompaniment,
            # but its second half can regress toward an unvoiced average while
            # backing continues to improve. Reuse the dedicated vocal head that
            # has always received vocal-only supervision instead of making the
            # two stems compete in one final linear projection. This keeps one
            # shared DiT and one sampling trajectory; only the last decoder head
            # is stem-specific.
            backing_out = out[..., : int(self.config.n_mels)]
            vocal_out = self.vocal_proj_out(x)
            if native_vocal_prior is not None:
                vocal_out = vocal_out + (
                    self.native_vocal_prior_strength() * native_vocal_prior
                )
            out = torch.cat((backing_out, vocal_out), dim=-1)
        elif native_vocal_prior is not None:
            out = out + self.native_vocal_prior_strength() * native_vocal_prior

        result: tuple = (out,)
        if repa_layer_idx is not None:
            repa_projected = self.repa_head(repa_hidden_raw) if repa_hidden_raw is not None else None
            result = result + (repa_projected,)
        if return_vocal_aux:
            # Shares the same AdaLN-modulated hidden state as the main head --
            # only the final linear projection differs (see __init__).
            result = result + (self.vocal_proj_out(x),)
        if return_native_vocal_prior:
            result = result + (native_vocal_prior,)
        if return_native_prosody:
            result = result + (native_prosody,)
        return result if len(result) > 1 else out
