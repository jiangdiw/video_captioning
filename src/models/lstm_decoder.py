# src/models/lstm_decoder.py
import torch
import torch.nn as nn

class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512,
                 num_layers=2, dropout=0.3, video_dim=512):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Project video embedding → LSTM initial hidden and cell state
        self.video_to_h = nn.Linear(video_dim, hidden_dim)
        self.video_to_c = nn.Linear(video_dim, hidden_dim)

        self.fc_out  = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def init_hidden(self, video_emb):
        """video_emb: (B, video_dim) → h0, c0"""
        h0 = self.video_to_h(video_emb)    # (B, hidden_dim)
        c0 = self.video_to_c(video_emb)    # (B, hidden_dim)
        # Stack for num_layers
        h0 = h0.unsqueeze(0).repeat(self.num_layers, 1, 1)
        c0 = c0.unsqueeze(0).repeat(self.num_layers, 1, 1)
        return h0, c0

    def forward(self, video_emb, captions):
        """
        Teacher forcing forward pass.
        video_emb : (B, video_dim)
        captions  : (B, max_len)    [SOS, w1, w2, ..., wN, EOS]
        returns   : logits (B, max_len-1, vocab_size)

        Input  to LSTM : captions[:, :-1] = [SOS, w1, ..., wN]
        Target of loss : captions[:, 1:]  = [w1, ..., wN, EOS]
        """
        inp    = captions[:, :-1]                      # (B, max_len-1)
        h0, c0 = self.init_hidden(video_emb)

        embeds = self.dropout(self.embedding(inp))     # (B, max_len-1, embed_dim)
        out, _ = self.lstm(embeds, (h0, c0))           # (B, max_len-1, hidden_dim)
        logits = self.fc_out(out)                      # (B, max_len-1, vocab_size)
        return logits

    @torch.no_grad()
    def generate(self, video_emb, max_len=20, sos_idx=1, eos_idx=2):
        """Greedy decoding at inference time."""
        B    = video_emb.size(0)
        h, c = self.init_hidden(video_emb)

        token     = torch.full((B, 1), sos_idx,
                               dtype=torch.long,
                               device=video_emb.device)
        generated = []

        for _ in range(max_len):
            embed        = self.embedding(token)          # (B, 1, embed_dim)
            out, (h, c)  = self.lstm(embed, (h, c))       # (B, 1, hidden_dim)
            logit        = self.fc_out(out.squeeze(1))    # (B, vocab_size)
            token        = logit.argmax(dim=-1, keepdim=True)  # (B, 1)
            generated.append(token)

        return torch.cat(generated, dim=1)                # (B, max_len)