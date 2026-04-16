# src/models/multimodal_encoder.py
import torch
import torch.nn as nn

# ================================================================
# APPROACH 1
# CLIP x DINOv2 cross-attention → mean pool → concat with audio
#
#   CLIP   (B, 40, 512) ──┐
#                          ├─► CrossAttn ─► mean pool ─► (B, 512) ──┐
#   DINOv2 (B, 40, 768) ──┘                                          ├─► concat ─► (B, 640)
#   Audio  (B,  T, 128) ────────────────── mean pool ─► (B, 128) ──┘
# ================================================================
class Approach1Encoder(nn.Module):
    def __init__(self, clip_dim=512, dino_dim=768, audio_dim=128, d_model=512, n_heads=8):
        super().__init__()

        # Project CLIP and DINOv2 to d_model
        self.clip_proj = nn.Linear(clip_dim, d_model)
        self.dino_proj = nn.Linear(dino_dim, d_model)

        # Cross-attention: CLIP as Query, DINOv2 as Key/Value
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)

        self.output_dim = d_model + audio_dim   # 512 + 128 = 640

    def forward(self, clip_emb, dino_emb, audio_emb):
        """
        clip_emb  : (B, 40, 512)
        dino_emb  : (B, 40, 768)
        audio_emb : (B,  T, 128)
        returns   : (B, 640)
        """
        # Project to common dim
        Q = self.clip_proj(clip_emb)    # (B, 40, 512)
        K = self.dino_proj(dino_emb)    # (B, 40, 512)
        V = K                           # same projection for K and V

        # Cross-attention + residual + norm
        attn_out, _ = self.cross_attn(Q, K, V)     # (B, 40, 512)
        visual_fused = self.norm(Q + attn_out)      # (B, 40, 512)

        # Mean pool over frames and time
        visual_pooled = visual_fused.mean(dim=1)    # (B, 512)
        audio_pooled  = audio_emb.mean(dim=1)       # (B, 128)

        # Concatenate
        out = torch.cat([visual_pooled, audio_pooled], dim=-1)  # (B, 640)
        return out


# ================================================================
# APPROACH 2
# Trimodal sequential cross-attention
#
#   Stage 1: CLIP (Q) x DINOv2 (K/V) ──► visual_fused (B, 40, d_model)
#   Stage 2: visual_fused (Q) x Audio (K/V) ──► trimodal (B, 40, d_model)
#   Mean pool ──► (B, d_model)
#
#   CLIP   (B, 40, 512) ──┐
#                          ├─► CrossAttn1 ─► visual_fused (B, 40, 512)
#   DINOv2 (B, 40, 768) ──┘        │
#                                   ├─► CrossAttn2 ─► mean pool ─► (B, 512)
#   Audio  (B,  T, 128) ───────────┘
# ================================================================
class Approach2Encoder(nn.Module):
    def __init__(self, clip_dim=512, dino_dim=768, audio_dim=128, d_model=512, n_heads=8):
        super().__init__()

        # Project each modality to d_model
        self.clip_proj  = nn.Linear(clip_dim,  d_model)
        self.dino_proj  = nn.Linear(dino_dim,  d_model)
        self.audio_proj = nn.Linear(audio_dim, d_model)

        # Stage 1: CLIP x DINOv2
        self.cross_attn1 = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)

        # Stage 2: visual_fused x Audio
        self.cross_attn2 = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)

        self.output_dim = d_model   # 512

    def forward(self, clip_emb, dino_emb, audio_emb):
        """
        clip_emb  : (B, 40, 512)
        dino_emb  : (B, 40, 768)
        audio_emb : (B,  T, 128)
        returns   : (B, 512)
        """
        # Project all to d_model
        clip_p  = self.clip_proj(clip_emb)      # (B, 40, 512)
        dino_p  = self.dino_proj(dino_emb)      # (B, 40, 512)
        audio_p = self.audio_proj(audio_emb)    # (B,  T, 512)

        # Stage 1: CLIP attends to DINOv2
        attn1, _     = self.cross_attn1(clip_p, dino_p, dino_p)    # (B, 40, 512)
        visual_fused = self.norm1(clip_p + attn1)                   # (B, 40, 512)

        # Stage 2: visual_fused attends to Audio
        attn2, _        = self.cross_attn2(visual_fused, audio_p, audio_p)  # (B, 40, 512)
        trimodal_fused  = self.norm2(visual_fused + attn2)                  # (B, 40, 512)

        # Mean pool over frames
        out = trimodal_fused.mean(dim=1)    # (B, 512)
        return out