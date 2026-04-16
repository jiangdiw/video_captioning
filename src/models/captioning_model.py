# src/models/captioning_model.py
import torch
import torch.nn as nn
from src.models.multimodal_encoder import Approach1Encoder, Approach2Encoder
from src.models.lstm_decoder import LSTMDecoder

class CaptioningModel(nn.Module):
    def __init__(self, encoder, vocab_size,
                 embed_dim=256, hidden_dim=512,
                 num_layers=2, dropout=0.3):
        super().__init__()

        self.encoder = encoder

        # Project encoder output → hidden_dim (makes decoder same for both approaches)
        self.proj = nn.Linear(encoder.output_dim, hidden_dim) \
                    if encoder.output_dim != hidden_dim \
                    else nn.Identity()

        self.decoder = LSTMDecoder(
            vocab_size  = vocab_size,
            embed_dim   = embed_dim,
            hidden_dim  = hidden_dim,
            num_layers  = num_layers,
            dropout     = dropout,
            video_dim   = hidden_dim   # always hidden_dim after projection
        )

    def forward(self, clip_emb, dino_emb, audio_emb, captions):
        """
        clip_emb  : (B, 40, 512)
        dino_emb  : (B, 40, 768)
        audio_emb : (B, T,  128)
        captions  : (B, max_len)
        returns   : logits (B, max_len-1, vocab_size)
        """
        video_emb = self.encoder(clip_emb, dino_emb, audio_emb)  # (B, encoder_dim)
        video_emb = self.proj(video_emb)                          # (B, hidden_dim)
        logits    = self.decoder(video_emb, captions)             # (B, max_len-1, vocab)
        return logits

    @torch.no_grad()
    def generate(self, clip_emb, dino_emb, audio_emb,
                 max_len=20, sos_idx=1, eos_idx=2):
        video_emb = self.encoder(clip_emb, dino_emb, audio_emb)
        video_emb = self.proj(video_emb)
        return self.decoder.generate(video_emb, max_len, sos_idx, eos_idx)