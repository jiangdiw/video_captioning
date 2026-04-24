# src/models/attention_lstm_decoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdditiveAttention(nn.Module):
    """
    Two-layer MLP attention (Bahdanau / additive attention)

    At each decoder step:
      score(h, e_i) = v · tanh(W · [h; e_i])
                           ↑              ↑
                         layer 2       layer 1

    h   : current LSTM hidden state  (B, hidden_dim)
    e_i : encoder output at frame i  (B, encoder_dim)

    Returns context vector = weighted sum of encoder outputs
    """
    def __init__(self, hidden_dim, encoder_dim):
        super().__init__()
        # Layer 1: project [hidden; encoder] → hidden_dim
        self.W = nn.Linear(hidden_dim + encoder_dim, hidden_dim)
        # Layer 2: project → scalar score (no bias so scores are symmetric)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_out):
        """
        hidden      : (B, hidden_dim)
        encoder_out : (B, 40, encoder_dim)
        returns:
            context : (B, encoder_dim)  weighted sum of frames
            weights : (B, 40)           attention weights (sum to 1)
        """
        T = encoder_out.size(1)   # 40 frames

        # Expand hidden to match all frames
        h_exp = hidden.unsqueeze(1).expand(-1, T, -1)  # (B, 40, hidden_dim)

        # Concatenate hidden with each frame
        combined = torch.cat([h_exp, encoder_out], dim=-1)  # (B, 40, hidden+encoder)

        # Layer 1: tanh(W * [h; e])
        energy  = torch.tanh(self.W(combined))   # (B, 40, hidden_dim)

        # Layer 2: v * energy → scalar score per frame
        scores  = self.v(energy).squeeze(-1)      # (B, 40)

        # Softmax → attention weights
        weights = F.softmax(scores, dim=-1)       # (B, 40)  sums to 1

        # Weighted sum of encoder outputs = context vector
        context = (weights.unsqueeze(-1) * encoder_out).sum(dim=1)  # (B, encoder_dim)

        return context, weights


class AttentionLSTMDecoder(nn.Module):
    """
    Attention-based LSTM decoder.

    At each time step t:
      1. Additive attention over encoder sequence → context vector c_t
      2. LSTM input = [word_embed(w_t) ; c_t]
      3. LSTMCell update → new hidden state h_t
      4. Predict next word: FC(h_t) → logits over vocab

    This forces the decoder to look back at the video
    at every single word it generates.
    """
    def __init__(self, vocab_size, embed_dim, hidden_dim,
                 encoder_dim, dropout=0.5):
        super().__init__()
        self.hidden_dim  = hidden_dim
        self.encoder_dim = encoder_dim

        self.embedding   = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.attention   = AdditiveAttention(hidden_dim, encoder_dim)

        # LSTMCell input = word embedding + context vector
        self.lstm_cell   = nn.LSTMCell(embed_dim + encoder_dim, hidden_dim)

        # Initialize cell state from hidden state
        self.init_c      = nn.Linear(hidden_dim, hidden_dim)

        self.fc_out      = nn.Linear(hidden_dim, vocab_size)
        self.dropout     = nn.Dropout(dropout)

    def forward(self, encoder_out, captions, init_h):
        """
        Teacher forcing forward pass.

        encoder_out : (B, 40, encoder_dim)   sequence to attend over
        captions    : (B, max_len)            [<sos>, w1, ..., wN, <eos>]
        init_h      : (B, hidden_dim)         from encoder pooled output

        Input  to LSTM: captions[:, :-1] = [<sos>, w1, ..., wN]
        Target of loss: captions[:, 1:]  = [w1, ..., wN, <eos>]

        returns: logits (B, max_len-1, vocab_size)
        """
        # Initialize LSTM states
        h = init_h                             # (B, hidden_dim)
        c = torch.tanh(self.init_c(init_h))   # (B, hidden_dim)

        # Teacher forcing: feed ground truth words as input
        inp    = captions[:, :-1]              # (B, max_len-1)
        embeds = self.dropout(
            self.embedding(inp))               # (B, max_len-1, embed_dim)

        logits = []
        for t in range(inp.size(1)):
            # Step 1: attention over 40 encoder frames
            context, attn_weights = self.attention(h, encoder_out)
            # context : (B, encoder_dim)
            # attn_weights : (B, 40)  ← tells us which frames mattered

            # Step 2: concat word embedding + context
            lstm_in = torch.cat(
                [embeds[:, t, :], context], dim=-1)  # (B, embed+encoder)

            # Step 3: LSTMCell update
            h, c = self.lstm_cell(lstm_in, (h, c))   # (B, hidden_dim)

            # Step 4: predict next word
            logit = self.fc_out(self.dropout(h))      # (B, vocab_size)
            logits.append(logit)

        return torch.stack(logits, dim=1)              # (B, max_len-1, vocab)

    @torch.no_grad()
    def generate(self, encoder_out, init_h,
                 max_len=20, sos_idx=1, eos_idx=2):
        """
        Greedy decoding with attention at each step.
        No teacher forcing — uses its own previous prediction.
        """
        B      = encoder_out.size(0)
        device = encoder_out.device

        h = init_h
        c = torch.tanh(self.init_c(init_h))

        token = torch.full(
            (B,), sos_idx, dtype=torch.long, device=device)
        generated = []

        for _ in range(max_len):
            # Attention over encoder sequence
            context, _ = self.attention(h, encoder_out)   # (B, encoder_dim)

            # Embed current token
            embed = self.embedding(token)                  # (B, embed_dim)

            # LSTM step
            lstm_in = torch.cat([embed, context], dim=-1)
            h, c    = self.lstm_cell(lstm_in, (h, c))

            # Predict next token
            logit = self.fc_out(h)                         # (B, vocab_size)
            token = logit.argmax(dim=-1)                   # (B,)
            generated.append(token)

            # Early stop if all sequences generated <eos>
            if (token == eos_idx).all():
                break

        return torch.stack(generated, dim=1)               # (B, max_len)