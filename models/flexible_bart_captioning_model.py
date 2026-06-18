import torch
import torch.nn as nn
from transformers import BartForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

from models.transfer_adapters import TargetAnchorConditionedResidualAdapter, TargetConditionedResidualAdapter


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
        max_audio_tokens: int | None = None,
    ) -> None:
        super().__init__()
        self.use_dino = use_dino
        self.use_audio = use_audio
        self.encoder_dim = d_model
        self.max_audio_tokens = max_audio_tokens

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

    def _compress_audio_tokens(
        self,
        audio_emb: torch.Tensor,
        audio_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.max_audio_tokens is None or audio_emb.shape[1] <= self.max_audio_tokens:
            return audio_emb, audio_mask

        batch_size, _, audio_dim = audio_emb.shape
        target_steps = self.max_audio_tokens
        pooled_audio = audio_emb.new_zeros(batch_size, target_steps, audio_dim)
        pooled_mask = audio_mask.new_zeros(batch_size, target_steps)

        for batch_index in range(batch_size):
            valid_steps = int(audio_mask[batch_index].sum().item())
            if valid_steps <= 0:
                continue
            if valid_steps <= target_steps:
                pooled_audio[batch_index, :valid_steps] = audio_emb[batch_index, :valid_steps]
                pooled_mask[batch_index, :valid_steps] = 1
                continue

            boundaries = torch.linspace(
                0,
                valid_steps,
                steps=target_steps + 1,
                device=audio_emb.device,
            ).floor().to(torch.long)
            for token_index in range(target_steps):
                start = int(boundaries[token_index].item())
                end = int(boundaries[token_index + 1].item())
                if end <= start:
                    end = min(start + 1, valid_steps)
                pooled_audio[batch_index, token_index] = audio_emb[batch_index, start:end].mean(dim=0)
                pooled_mask[batch_index, token_index] = 1

        return pooled_audio, pooled_mask

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
            audio_emb, audio_mask = self._compress_audio_tokens(audio_emb, audio_mask)
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
        decoder_train_mode: str = "full",
        decoder_train_last_n_layers: int = 2,
        gradient_checkpointing: bool = True,
        max_audio_tokens: int | None = None,
        transfer_adapter: str = "none",
        transfer_num_anchors: int = 0,
        transfer_num_tokens: int = 4,
        transfer_bottleneck_dim: int = 64,
        transfer_dropout: float = 0.1,
        transfer_gate_init: float = -4.0,
        transfer_residual_scale: float = 1.0,
        transfer_anchor_temperature: float = 0.08,
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
            max_audio_tokens=max_audio_tokens,
        )

        self.bart = BartForConditionalGeneration.from_pretrained(bart_model_name)
        self.bart.config.use_cache = False
        if gradient_checkpointing:
            self.bart.gradient_checkpointing_enable()
        self.proj = nn.Linear(encoder_d_model, self.bart.config.d_model)
        self.transfer_adapter_name = transfer_adapter.lower()
        if self.transfer_adapter_name == "none":
            self.transfer_adapter = None
        elif self.transfer_adapter_name == "target_residual_prefix":
            self.transfer_adapter = TargetConditionedResidualAdapter(
                d_model=self.bart.config.d_model,
                n_heads=n_heads,
                num_prefix_tokens=transfer_num_tokens,
                bottleneck_dim=transfer_bottleneck_dim,
                dropout=transfer_dropout,
                gate_init=transfer_gate_init,
                residual_scale=transfer_residual_scale,
            )
        elif self.transfer_adapter_name == "target_anchor_prefix":
            self.transfer_adapter = TargetAnchorConditionedResidualAdapter(
                d_model=self.bart.config.d_model,
                n_heads=n_heads,
                num_anchors=transfer_num_anchors,
                num_prefix_tokens=transfer_num_tokens,
                bottleneck_dim=transfer_bottleneck_dim,
                dropout=transfer_dropout,
                gate_init=transfer_gate_init,
                residual_scale=transfer_residual_scale,
                anchor_temperature=transfer_anchor_temperature,
            )
        else:
            raise ValueError(f"Unsupported transfer_adapter: {transfer_adapter}")

        self._configure_decoder_train_mode(
            decoder_train_mode=decoder_train_mode,
            decoder_train_last_n_layers=decoder_train_last_n_layers,
        )

    def _freeze_all_bart(self) -> None:
        for parameter in self.bart.parameters():
            parameter.requires_grad = False

    def _configure_decoder_train_mode(
        self,
        decoder_train_mode: str,
        decoder_train_last_n_layers: int,
    ) -> None:
        if decoder_train_mode == "full":
            return

        self._freeze_all_bart()
        if decoder_train_mode == "freeze":
            return
        if decoder_train_mode != "partial":
            raise ValueError(f"Unsupported decoder_train_mode: {decoder_train_mode}")

        decoder = self.bart.model.decoder
        if decoder_train_last_n_layers > 0:
            last_n = min(decoder_train_last_n_layers, len(decoder.layers))
            for layer in decoder.layers[-last_n:]:
                for parameter in layer.parameters():
                    parameter.requires_grad = True

        # Keep tied language modeling weights trainable for limited lexical adaptation.
        self.bart.model.shared.weight.requires_grad = True
        decoder.embed_positions.weight.requires_grad = True
        for module_name in ("layernorm_embedding", "layer_norm"):
            module = getattr(decoder, module_name, None)
            if module is None:
                continue
            for parameter in module.parameters():
                parameter.requires_grad = True
        for parameter in self.bart.lm_head.parameters():
            parameter.requires_grad = True

    def set_transfer_anchor_bank(self, anchor_embeddings: torch.Tensor) -> None:
        if self.transfer_adapter is None or not hasattr(self.transfer_adapter, "set_anchor_embeddings"):
            raise ValueError("The active transfer adapter does not accept an anchor bank")
        self.transfer_adapter.set_anchor_embeddings(anchor_embeddings)

    def _encode(
        self,
        clip_emb: torch.Tensor,
        dino_emb: torch.Tensor | None,
        audio_emb: torch.Tensor | None,
        audio_mask: torch.Tensor | None,
        transfer_anchor_targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        resolved_audio_mask = _infer_audio_mask(audio_emb, audio_mask)
        sequence, attention_mask = self.encoder(
            clip_emb,
            dino_emb,
            audio_emb,
            audio_mask=resolved_audio_mask,
        )
        encoder_hidden_states = self.proj(sequence)
        if self.transfer_adapter is not None:
            if isinstance(self.transfer_adapter, TargetAnchorConditionedResidualAdapter):
                encoder_hidden_states, attention_mask = self.transfer_adapter(
                    encoder_hidden_states,
                    attention_mask,
                    anchor_targets=transfer_anchor_targets,
                )
            else:
                encoder_hidden_states, attention_mask = self.transfer_adapter(encoder_hidden_states, attention_mask)
        return encoder_hidden_states, attention_mask

    def forward(
        self,
        clip_emb: torch.Tensor,
        dino_emb: torch.Tensor | None = None,
        audio_emb: torch.Tensor | None = None,
        audio_mask: torch.Tensor | None = None,
        decoder_input_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        transfer_anchor_targets: torch.Tensor | None = None,
    ):
        encoder_hidden_states, attention_mask = self._encode(
            clip_emb,
            dino_emb,
            audio_emb,
            audio_mask,
            transfer_anchor_targets=transfer_anchor_targets,
        )
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
        transfer_anchor_targets: torch.Tensor | None = None,
        max_new_tokens: int = 40,
        **kwargs,
    ) -> torch.Tensor:
        encoder_hidden_states, attention_mask = self._encode(
            clip_emb,
            dino_emb,
            audio_emb,
            audio_mask,
            transfer_anchor_targets=transfer_anchor_targets,
        )
        return self.bart.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=encoder_hidden_states),
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )
