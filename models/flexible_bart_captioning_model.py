import torch
import torch.nn as nn
from transformers import BartForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput


def _infer_audio_mask(audio_emb: torch.Tensor | None, audio_mask: torch.Tensor | None) -> torch.Tensor | None:
    if audio_emb is None:
        return None
    if audio_mask is not None:
        return audio_mask.to(device=audio_emb.device, dtype=torch.long)
    return (audio_emb.abs().sum(dim=-1) > 0).long()


class FlexibleVisualEncoder(nn.Module):
    def __init__(
        self,
        use_dino: bool,
        use_audio: bool,
        clip_dim: int = 512,
        dino_dim: int = 768,
        audio_dim: int = 128,
        d_model: int = 512,
        n_heads: int = 8,
    ) -> None:
        super().__init__()
        self.use_dino = use_dino
        self.use_audio = use_audio
        self.encoder_dim = d_model

        self.clip_proj = nn.Linear(clip_dim, d_model)
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
        else:
            self.audio_proj = None

    def forward(
        self,
        clip_emb: torch.Tensor,
        dino_emb: torch.Tensor | None = None,
        audio_emb: torch.Tensor | None = None,
        audio_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        clip_tokens = self.clip_proj(clip_emb)
        clip_self_out, _ = self.clip_self_attn(clip_tokens, clip_tokens, clip_tokens)
        visual_seq = self.clip_self_norm(clip_tokens + clip_self_out)
        attention_mask = torch.ones(
            visual_seq.shape[:2],
            dtype=torch.long,
            device=visual_seq.device,
        )

        if self.use_dino:
            if dino_emb is None:
                raise ValueError("dino_emb is required when use_dino=True")
            dino_tokens = self.dino_proj(dino_emb)
            cross_out, _ = self.cross_attn(visual_seq, dino_tokens, dino_tokens)
            visual_seq = self.cross_norm(visual_seq + cross_out)

        if self.use_audio:
            if audio_emb is None:
                raise ValueError("audio_emb is required when use_audio=True")
            if audio_mask is None:
                audio_mask = torch.ones(
                    audio_emb.shape[:2],
                    dtype=torch.long,
                    device=audio_emb.device,
                )
            audio_tokens = self.audio_proj(audio_emb)
            audio_tokens = audio_tokens * audio_mask.unsqueeze(-1).to(audio_tokens.dtype)
            visual_seq = torch.cat([visual_seq, audio_tokens], dim=1)
            attention_mask = torch.cat(
                [attention_mask, audio_mask.to(dtype=torch.long, device=attention_mask.device)],
                dim=1,
            )

        return visual_seq, attention_mask


class FlexibleBartCaptioningModel(nn.Module):
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
        freeze_decoder: bool = False,
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        self.use_dino = use_dino
        self.use_audio = use_audio
        self.encoder = FlexibleVisualEncoder(
            use_dino=use_dino,
            use_audio=use_audio,
            clip_dim=clip_dim,
            dino_dim=dino_dim,
            audio_dim=audio_dim,
            d_model=encoder_d_model,
            n_heads=n_heads,
        )

        self.bart = BartForConditionalGeneration.from_pretrained(bart_model_name)
        self.bart.config.use_cache = False
        if gradient_checkpointing:
            self.bart.gradient_checkpointing_enable()
        self.proj = nn.Linear(encoder_d_model, self.bart.config.d_model)

        if freeze_decoder:
            # In the stable path, BART is only used as a fixed pretrained decoder stack
            # over external encoder outputs. Freeze the entire BART module explicitly so
            # optimizer state and LR settings only affect the external encoder + proj.
            for parameter in self.bart.parameters():
                parameter.requires_grad = False

    def _encode(
        self,
        clip_emb: torch.Tensor,
        dino_emb: torch.Tensor | None,
        audio_emb: torch.Tensor | None,
        audio_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        resolved_audio_mask = _infer_audio_mask(audio_emb, audio_mask)
        sequence, attention_mask = self.encoder(
            clip_emb,
            dino_emb,
            audio_emb,
            audio_mask=resolved_audio_mask,
        )
        return self.proj(sequence), attention_mask

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
