import torch
import torch.nn as nn
from transformers import BartForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

from models.multimodal_encoder_40embedding import Approach1Encoder


class BartCaptioningModelNoAudio(nn.Module):
    def __init__(self,
                 clip_dim=512, dino_dim=768,
                 encoder_d_model=512, n_heads=8,
                 bart_model_name="facebook/bart-base",
                 freeze_bart=False):
        super().__init__()

        # audio_dim kept at default so Approach1Encoder initialises normally;
        # audio is never passed in forward — encoder ignores it anyway.
        self.encoder = Approach1Encoder(
            clip_dim=clip_dim,
            dino_dim=dino_dim,
            d_model=encoder_d_model,
            n_heads=n_heads,
        )

        self.bart = BartForConditionalGeneration.from_pretrained(bart_model_name)
        bart_d_model = self.bart.config.d_model

        self.proj = nn.Linear(encoder_d_model, bart_d_model)

        if freeze_bart:
            for p in self.bart.parameters():
                p.requires_grad = False
            for p in self.bart.lm_head.parameters():
                p.requires_grad = True

    def _encode(self, clip_emb, dino_emb):
        dummy_audio = torch.zeros(
            clip_emb.size(0), 1, 128, device=clip_emb.device
        )
        seq = self.encoder(clip_emb, dino_emb, dummy_audio)  # (B, 40, 512)
        return self.proj(seq)                                 # (B, 40, 768)

    def forward(self, clip_emb, dino_emb, decoder_input_ids=None, labels=None):
        """
        clip_emb : (B, 40, 512)
        dino_emb : (B, 40, 768)
        labels   : (B, seq_len) with -100 for padding
        """
        encoder_hidden_states = self._encode(clip_emb, dino_emb)
        attention_mask = torch.ones(
            encoder_hidden_states.shape[:2],
            dtype=torch.long,
            device=encoder_hidden_states.device,
        )
        return self.bart(
            attention_mask=attention_mask,
            encoder_outputs=(encoder_hidden_states,),
            decoder_input_ids=decoder_input_ids,
            labels=labels,
        )

    @torch.no_grad()
    def generate(self, clip_emb, dino_emb, max_new_tokens=40, **kwargs):
        encoder_hidden_states = self._encode(clip_emb, dino_emb)
        attention_mask = torch.ones(
            encoder_hidden_states.shape[:2],
            dtype=torch.long,
            device=encoder_hidden_states.device,
        )
        return self.bart.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=encoder_hidden_states),
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )
