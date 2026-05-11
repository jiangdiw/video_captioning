import torch
import torch.nn as nn
from transformers import BartForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

from models.multimodal_encoder_40embedding import Approach1Encoder


class BartCaptioningModel(nn.Module):
    def __init__(self,
                 clip_dim=512, dino_dim=768, audio_dim=128,
                 encoder_d_model=512, n_heads=8,
                 bart_model_name="facebook/bart-base",
                 freeze_bart=False):
        super().__init__()

        self.encoder = Approach1Encoder(
            clip_dim=clip_dim,
            dino_dim=dino_dim,
            audio_dim=audio_dim,
            d_model=encoder_d_model,
            n_heads=n_heads,
        )

        self.bart = BartForConditionalGeneration.from_pretrained(bart_model_name)
        bart_d_model = self.bart.config.d_model   # 768 for bart-base

        # Project encoder sequence → BART hidden size
        self.proj = nn.Linear(encoder_d_model, bart_d_model)

        if freeze_bart:
            for p in self.bart.parameters():
                p.requires_grad = False
            # Always keep lm_head trainable
            for p in self.bart.lm_head.parameters():
                p.requires_grad = True

    def _resolve_audio_mask(self, audio_emb, audio_mask=None):
        if audio_emb is None:
            return None
        if audio_mask is not None:
            return audio_mask.to(device=audio_emb.device, dtype=torch.long)
        return (audio_emb.abs().sum(dim=-1) > 0).long()

    def _encode(self, clip_emb, dino_emb, audio_emb, audio_mask=None):
        """Returns projected encoder sequence + attention mask."""
        resolved_audio_mask = self._resolve_audio_mask(audio_emb, audio_mask)
        seq = self.encoder(clip_emb, dino_emb, audio_emb, audio_mask=resolved_audio_mask)
        encoder_hidden_states = self.proj(seq)

        attention_mask = torch.ones(
            encoder_hidden_states.shape[:2],
            dtype=torch.long,
            device=encoder_hidden_states.device,
        )
        visual_len = clip_emb.shape[1]
        if resolved_audio_mask is not None:
            attention_mask[:, visual_len:] = resolved_audio_mask
        return encoder_hidden_states, attention_mask

    def forward(self, clip_emb, dino_emb, audio_emb, audio_mask=None, decoder_input_ids=None, labels=None):
        """
        clip_emb          : (B, 40, 512)
        dino_emb          : (B, 40, 768)
        audio_emb         : (B,  T, 128)
        decoder_input_ids : (B, seq_len)   — BART token ids, shifted right
        labels            : (B, seq_len)   — token ids with -100 for padding
        returns: transformers CausalLMOutputWithCrossAttentions
        """
        encoder_hidden_states, attention_mask = self._encode(clip_emb, dino_emb, audio_emb, audio_mask=audio_mask)

        return self.bart(
            attention_mask=attention_mask,
            encoder_outputs=(encoder_hidden_states,),
            decoder_input_ids=decoder_input_ids,
            labels=labels,
        )

    @torch.no_grad()
    def generate(self, clip_emb, dino_emb, audio_emb, audio_mask=None, max_new_tokens=40, **kwargs):
        """
        Returns generated token id tensors (B, seq_len).
        Extra kwargs are forwarded to bart.generate (e.g. num_beams).
        """
        encoder_hidden_states, attention_mask = self._encode(clip_emb, dino_emb, audio_emb, audio_mask=audio_mask)

        return self.bart.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=encoder_hidden_states),
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )
