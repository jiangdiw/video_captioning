# src/models/multimodal_encoder.py
import torch
import torch.nn as nn

# ================================================================
# APPROACH 1
# CLIP (Q) x DINOv2 (K/V) cross-attention
# Audio projected to d_model and appended to visual sequence
# Returns sequence (B, 40+T, d_model) for BART decoder cross-attention
# ================================================================
class Approach1Encoder(nn.Module):
    def __init__(self, clip_dim=512, dino_dim=768,
                 audio_dim=128, d_model=512, n_heads=8):
        super().__init__()
        self.clip_proj  = nn.Linear(clip_dim,  d_model)
        self.dino_proj  = nn.Linear(dino_dim,  d_model)
        self.audio_proj = nn.Linear(audio_dim, d_model)

        self.clip_self_attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True)
        self.clip_self_norm = nn.LayerNorm(d_model)

        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True)
        self.norm        = nn.LayerNorm(d_model)

        self.encoder_dim = d_model  # sequence dim for cross-attention

    def forward(self, clip_emb, dino_emb, audio_emb=None, audio_mask=None):
        """
        clip_emb  : (B, 40, 512)
        dino_emb  : (B, 40, 768)
        audio_emb : (B,  T, 128) or None — if None, audio is skipped
        returns:
            seq : (B, 40+T, 512) with audio, or (B, 40, 512) without
        """
        Q = self.clip_proj(clip_emb)     # (B, 40, 512)
        K = self.dino_proj(dino_emb)     # (B, 40, 512)
        V = K

        clip_self_out, _ = self.clip_self_attn(Q, Q, Q)
        Q = self.clip_self_norm(Q + clip_self_out)

        attn_out, _ = self.cross_attn(Q, K, V)
        visual_seq = self.norm(Q + attn_out)          # (B, 40, 512)

        if audio_emb is None:
            return visual_seq                         # (B, 40, 512)

        audio_seq = self.audio_proj(audio_emb)        # (B,  T, 512)
        if audio_mask is None:
            audio_mask = torch.ones(
                audio_emb.shape[:2],
                dtype=audio_seq.dtype,
                device=audio_seq.device,
            )
        else:
            audio_mask = audio_mask.to(dtype=audio_seq.dtype, device=audio_seq.device)
        audio_seq = audio_seq * audio_mask.unsqueeze(-1)
        return torch.cat([visual_seq, audio_seq], dim=1)  # (B, 40+T, 512)


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

        self.clip_self_attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True)
        self.clip_self_norm = nn.LayerNorm(d_model)

        self.cross_attn1 = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True)
        self.norm1        = nn.LayerNorm(d_model)

        self.cross_attn2 = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True)
        self.norm2        = nn.LayerNorm(d_model)

        self.encoder_dim = d_model  # sequence dim for cross-attention

    def forward(self, clip_emb, dino_emb, audio_emb):
        """
        clip_emb  : (B, 40, 512)
        dino_emb  : (B, 40, 768)
        audio_emb : (B,  T, 128)
        returns:
            seq : (B, 40, 512)
        """
        clip_p  = self.clip_proj(clip_emb)      # (B, 40, 512)
        dino_p  = self.dino_proj(dino_emb)      # (B, 40, 512)
        audio_p = self.audio_proj(audio_emb)    # (B,  T, 512)

        # CLIP self-attention
        clip_self_out, _ = self.clip_self_attn(clip_p, clip_p, clip_p)
        clip_p = self.clip_self_norm(clip_p + clip_self_out)  # (B, 40, 512)

        # Stage 1: CLIP attends to DINOv2
        attn1, _     = self.cross_attn1(clip_p, dino_p, dino_p)
        visual_fused = self.norm1(clip_p + attn1)             # (B, 40, 512)

        # Stage 2: visual attends to Audio
        attn2, _ = self.cross_attn2(visual_fused, audio_p, audio_p)
        seq = self.norm2(visual_fused + attn2)                # (B, 40, 512)

        return seq
