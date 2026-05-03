# src/models/multimodal_encoder.py
import torch
import torch.nn as nn

# ================================================================
# APPROACH 1
# CLIP (Q) x DINOv2 (K/V) cross-attention
# Returns sequence (B, 40, d_model) for attention decoder
# Audio mean-pooled for LSTM initialization
# ================================================================
class Approach1Encoder(nn.Module):
    def __init__(self, clip_dim=512, dino_dim=768,
                 audio_dim=128, d_model=512, n_heads=8):
        super().__init__()
        self.clip_proj  = nn.Linear(clip_dim,  d_model)
        self.dino_proj  = nn.Linear(dino_dim,  d_model)

        self.clip_self_attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True)
        self.clip_self_norm = nn.LayerNorm(d_model)

        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True)

        self.norm        = nn.LayerNorm(d_model)

        self.encoder_dim = d_model              # sequence dim for attention
        self.init_dim    = d_model + audio_dim  # 512+128=640 for LSTM init

    def forward(self, clip_emb, dino_emb, audio_emb):
        """
        clip_emb  : (B, 40, 512)
        dino_emb  : (B, 40, 768)
        audio_emb : (B,  T, 128)
        returns:
            seq    : (B, 40, 512)  visual sequence for attention
            pooled : (B, 640)      visual+audio pooled for LSTM init
        """
        Q = self.clip_proj(clip_emb)     # (B, 40, 512)
        K = self.dino_proj(dino_emb)     # (B, 40, 512)
        V = K

        clip_self_out, _ = self.clip_self_attn(Q, Q, Q)
        Q = self.clip_self_norm(Q + clip_self_out)

        attn_out, _ = self.cross_attn(Q, K, V)
        seq = self.norm(Q + attn_out)

        # For LSTM initialization: pool visual + audio
        visual_pooled = seq.mean(dim=1)            # (B, 512)
        audio_pooled  = audio_emb.mean(dim=1)      # (B, 128)
        pooled = torch.cat([visual_pooled,
                            audio_pooled], dim=-1) # (B, 640)

        return seq, pooled


# ================================================================
# APPROACH 2
# Stage 1: CLIP (Q) x DINOv2 (K/V)
# Stage 2: visual_fused (Q) x Audio (K/V)
# Returns sequence (B, 40, d_model) — audio context baked into each frame
# ================================================================
class Approach2Encoder(nn.Module):
    def __init__(self, clip_dim=512, dino_dim=768,
                 audio_dim=128, d_model=512, n_heads=8):
        super().__init__()
        self.clip_proj   = nn.Linear(clip_dim,  d_model)
        self.dino_proj   = nn.Linear(dino_dim,  d_model)
        self.audio_proj  = nn.Linear(audio_dim, d_model)

        self.cross_attn1 = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True)
        self.norm1        = nn.LayerNorm(d_model)

        self.cross_attn2 = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True)
        self.norm2        = nn.LayerNorm(d_model)

        self.encoder_dim = d_model   # sequence dim for attention
        self.init_dim    = d_model   # 512 for LSTM init

    def forward(self, clip_emb, dino_emb, audio_emb):
        """
        clip_emb  : (B, 40, 512)
        dino_emb  : (B, 40, 768)
        audio_emb : (B,  T, 128)
        returns:
            seq    : (B, 40, 512)  trimodal sequence for attention
            pooled : (B, 512)      pooled for LSTM init
        """
        clip_p  = self.clip_proj(clip_emb)      # (B, 40, 512)
        dino_p  = self.dino_proj(dino_emb)      # (B, 40, 512)
        audio_p = self.audio_proj(audio_emb)    # (B,  T, 512)

        # Stage 1: CLIP attends to DINOv2
        attn1, _     = self.cross_attn1(clip_p, dino_p, dino_p)
        visual_fused = self.norm1(clip_p + attn1)            # (B, 40, 512)

        # Stage 2: visual attends to Audio
        attn2, _ = self.cross_attn2(visual_fused, audio_p, audio_p)
        seq = self.norm2(visual_fused + attn2)               # (B, 40, 512)

        pooled = seq.mean(dim=1)   # (B, 512)

        return seq, pooled