# src/models/captioning_model.py
import torch
import torch.nn as nn
from src.models.attention_lstm_decoder import AttentionLSTMDecoder


class CaptioningModel(nn.Module):
    def __init__(self, encoder, vocab_size,
                 embed_dim=256, hidden_dim=512, dropout=0.5):
        super().__init__()
        self.encoder = encoder

        # Project encoder pooled output → hidden_dim for LSTM init
        self.init_proj = nn.Linear(encoder.init_dim, hidden_dim)

        self.decoder = AttentionLSTMDecoder(
            vocab_size   = vocab_size,
            embed_dim    = embed_dim,
            hidden_dim   = hidden_dim,
            encoder_dim  = encoder.encoder_dim,
            dropout      = dropout,
        )

    def forward(self, clip_emb, dino_emb, audio_emb, captions):
        """
        clip_emb  : (B, 40, 512)
        dino_emb  : (B, 40, 768)
        audio_emb : (B,  T, 128)
        captions  : (B, max_len)
        returns   : logits (B, max_len-1, vocab_size)
        """
        # Encoder: returns sequence for attention + pooled for init
        seq, pooled = self.encoder(clip_emb, dino_emb, audio_emb)
        # seq    : (B, 40, encoder_dim)
        # pooled : (B, init_dim)

        # Project pooled → hidden_dim for LSTM initialization
        init_h = torch.tanh(self.init_proj(pooled))   # (B, hidden_dim)

        # Attention LSTM decoder
        logits = self.decoder(seq, captions, init_h)  # (B, max_len-1, vocab)
        return logits

    @torch.no_grad()
    def generate(self, clip_emb, dino_emb, audio_emb,
                 max_len=20, sos_idx=1, eos_idx=2):
        seq, pooled = self.encoder(clip_emb, dino_emb, audio_emb)
        init_h      = torch.tanh(self.init_proj(pooled))
        return self.decoder.generate(
            seq, init_h, max_len, sos_idx, eos_idx)