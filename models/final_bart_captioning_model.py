import torch
import torch.nn as nn
from transformers import BartForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput


def infer_audio_mask(audio_emb: torch.Tensor | None, audio_mask: torch.Tensor | None) -> torch.Tensor | None:
    if audio_emb is None:
        return None
    if audio_mask is not None:
        return audio_mask.to(device=audio_emb.device, dtype=torch.long)
    return (audio_emb.abs().sum(dim=-1) > 0).long()


class FinalVisualEncoder(nn.Module):
    def __init__(
        self,
        use_dino: bool,
        use_audio: bool,
        clip_dim: int = 512,
        dino_dim: int = 768,
        audio_dim: int = 128,
        d_model: int = 512,
        n_heads: int = 8,
        max_visual_positions: int = 40,
        audio_summary_tokens: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.use_dino = use_dino
        self.use_audio = use_audio
        self.max_visual_positions = max_visual_positions
        self.audio_summary_tokens = audio_summary_tokens
        self.encoder_dim = d_model

        self.clip_proj = nn.Linear(clip_dim, d_model)
        self.clip_pos_embed = nn.Embedding(max_visual_positions, d_model)
        self.clip_self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.clip_self_norm = nn.LayerNorm(d_model)

        if use_dino:
            self.dino_proj = nn.Linear(dino_dim, d_model)
            self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
            self.cross_norm = nn.LayerNorm(d_model)
        else:
            self.dino_proj = None
            self.cross_attn = None
            self.cross_norm = None

        if use_audio:
            self.audio_proj = nn.Linear(audio_dim, d_model)
            self.audio_queries = nn.Parameter(torch.randn(audio_summary_tokens, d_model) * 0.02)
            self.audio_cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
            self.audio_norm = nn.LayerNorm(d_model)
        else:
            self.audio_proj = None
            self.audio_queries = None
            self.audio_cross_attn = None
            self.audio_norm = None

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        clip_emb: torch.Tensor,
        dino_emb: torch.Tensor | None = None,
        audio_emb: torch.Tensor | None = None,
        audio_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        batch_size, visual_len, _ = clip_emb.shape
        if visual_len > self.max_visual_positions:
            raise ValueError(
                f"Visual sequence length {visual_len} exceeds max_visual_positions={self.max_visual_positions}"
            )

        positions = torch.arange(visual_len, device=clip_emb.device).unsqueeze(0).expand(batch_size, -1)
        clip_tokens = self.clip_proj(clip_emb) + self.clip_pos_embed(positions)
        clip_tokens = self.dropout(clip_tokens)
        clip_self_out, _ = self.clip_self_attn(clip_tokens, clip_tokens, clip_tokens)
        visual_seq = self.clip_self_norm(clip_tokens + clip_self_out)

        if self.use_dino:
            if dino_emb is None:
                raise ValueError("dino_emb is required when use_dino=True")
            dino_tokens = self.dino_proj(dino_emb) + self.clip_pos_embed(positions)
            dino_tokens = self.dropout(dino_tokens)
            cross_out, _ = self.cross_attn(visual_seq, dino_tokens, dino_tokens)
            visual_seq = self.cross_norm(visual_seq + cross_out)

        attention_mask = torch.ones(batch_size, visual_len, dtype=torch.long, device=clip_emb.device)
        audio_summary_len = 0

        if self.use_audio:
            if audio_emb is None:
                raise ValueError("audio_emb is required when use_audio=True")
            resolved_audio_mask = infer_audio_mask(audio_emb, audio_mask)
            if resolved_audio_mask is None:
                raise ValueError("audio_mask resolution failed")

            audio_tokens = self.audio_proj(audio_emb)
            audio_tokens = self.dropout(audio_tokens)

            valid_audio = resolved_audio_mask.bool()
            safe_valid_audio = valid_audio.clone()
            no_audio_rows = ~safe_valid_audio.any(dim=1)
            if no_audio_rows.any():
                safe_valid_audio[no_audio_rows, 0] = True
                audio_tokens = audio_tokens.clone()
                audio_tokens[no_audio_rows, 0] = 0.0

            query = self.audio_queries.unsqueeze(0).expand(batch_size, -1, -1)
            audio_out, _ = self.audio_cross_attn(
                query,
                audio_tokens,
                audio_tokens,
                key_padding_mask=~safe_valid_audio,
            )
            audio_summary = self.audio_norm(query + audio_out)
            audio_summary = audio_summary * valid_audio.any(dim=1, keepdim=True).unsqueeze(-1).to(audio_summary.dtype)

            visual_seq = torch.cat([visual_seq, audio_summary], dim=1)
            audio_summary_len = self.audio_summary_tokens
            audio_summary_mask = valid_audio.any(dim=1, keepdim=True).long().expand(-1, self.audio_summary_tokens)
            attention_mask = torch.cat([attention_mask, audio_summary_mask], dim=1)

        return visual_seq, attention_mask, audio_summary_len


class FinalBartCaptioningModel(nn.Module):
    def __init__(
        self,
        use_dino: bool,
        use_audio: bool,
        clip_dim: int = 512,
        dino_dim: int = 768,
        audio_dim: int = 128,
        encoder_d_model: int = 512,
        n_heads: int = 8,
        bart_model_name: str = "facebook/bart-base",
        decoder_train_mode: str = "partial",
        decoder_train_last_n_layers: int = 2,
        max_visual_positions: int = 40,
        audio_summary_tokens: int = 4,
        feature_dropout: float = 0.1,
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        self.use_dino = use_dino
        self.use_audio = use_audio
        self.encoder = FinalVisualEncoder(
            use_dino=use_dino,
            use_audio=use_audio,
            clip_dim=clip_dim,
            dino_dim=dino_dim,
            audio_dim=audio_dim,
            d_model=encoder_d_model,
            n_heads=n_heads,
            max_visual_positions=max_visual_positions,
            audio_summary_tokens=audio_summary_tokens,
            dropout=feature_dropout,
        )

        self.bart = BartForConditionalGeneration.from_pretrained(bart_model_name)
        self.bart.config.use_cache = False
        if gradient_checkpointing:
            self.bart.gradient_checkpointing_enable()
        bart_dim = self.bart.config.d_model
        self.proj = nn.Linear(encoder_d_model, bart_dim)
        self.input_norm = nn.LayerNorm(bart_dim)
        self.feature_dropout = nn.Dropout(feature_dropout)
        self.modality_type_embeddings = nn.Embedding(2, bart_dim)

        self._configure_decoder_tuning(
            decoder_train_mode=decoder_train_mode,
            decoder_train_last_n_layers=decoder_train_last_n_layers,
        )

    def _configure_decoder_tuning(self, decoder_train_mode: str, decoder_train_last_n_layers: int) -> None:
        mode = decoder_train_mode.lower()
        if mode not in {"full", "freeze", "partial"}:
            raise ValueError(f"Unsupported decoder_train_mode: {decoder_train_mode}")
        if mode == "full":
            return

        for parameter in self.bart.parameters():
            parameter.requires_grad = False

        # Keep output embedding trainable for final adaptation.
        self.bart.model.shared.weight.requires_grad = True
        for parameter in self.bart.lm_head.parameters():
            parameter.requires_grad = True

        if mode == "freeze":
            return

        decoder = self.bart.model.decoder
        for parameter in decoder.layernorm_embedding.parameters():
            parameter.requires_grad = True
        for parameter in decoder.embed_positions.parameters():
            parameter.requires_grad = True

        if hasattr(decoder, "layer_norm") and decoder.layer_norm is not None:
            for parameter in decoder.layer_norm.parameters():
                parameter.requires_grad = True

        last_n = max(1, min(decoder_train_last_n_layers, len(decoder.layers)))
        for layer in decoder.layers[-last_n:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True

    def _encode(
        self,
        clip_emb: torch.Tensor,
        dino_emb: torch.Tensor | None,
        audio_emb: torch.Tensor | None,
        audio_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sequence, attention_mask, audio_summary_len = self.encoder(
            clip_emb,
            dino_emb,
            audio_emb,
            audio_mask=audio_mask,
        )
        encoder_hidden_states = self.feature_dropout(self.input_norm(self.proj(sequence)))
        token_type_ids = torch.zeros(
            encoder_hidden_states.shape[:2],
            dtype=torch.long,
            device=encoder_hidden_states.device,
        )
        if audio_summary_len > 0:
            token_type_ids[:, -audio_summary_len:] = 1
        encoder_hidden_states = encoder_hidden_states + self.modality_type_embeddings(token_type_ids)
        return encoder_hidden_states, attention_mask

    def forward(
        self,
        clip_emb: torch.Tensor,
        dino_emb: torch.Tensor | None = None,
        audio_emb: torch.Tensor | None = None,
        audio_mask: torch.Tensor | None = None,
        decoder_input_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ):
        encoder_hidden_states, attention_mask = self._encode(clip_emb, dino_emb, audio_emb, audio_mask)
        return self.bart(
            attention_mask=attention_mask,
            encoder_outputs=(encoder_hidden_states,),
            decoder_input_ids=decoder_input_ids,
            labels=labels,
        )

    @torch.no_grad()
    def generate(
        self,
        clip_emb: torch.Tensor,
        dino_emb: torch.Tensor | None = None,
        audio_emb: torch.Tensor | None = None,
        audio_mask: torch.Tensor | None = None,
        max_new_tokens: int = 40,
        **kwargs,
    ) -> torch.Tensor:
        encoder_hidden_states, attention_mask = self._encode(clip_emb, dino_emb, audio_emb, audio_mask)
        return self.bart.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=encoder_hidden_states),
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )
