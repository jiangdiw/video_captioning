import torch
import torch.nn as nn
import torch.nn.functional as F


class TargetConditionedResidualAdapter(nn.Module):
    """Small target-domain adapter for visual encoder states before BART decoding."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        num_prefix_tokens: int = 4,
        bottleneck_dim: int = 64,
        dropout: float = 0.1,
        gate_init: float = -4.0,
        residual_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if bottleneck_dim <= 0:
            raise ValueError("bottleneck_dim must be positive")
        if num_prefix_tokens < 0:
            raise ValueError("num_prefix_tokens must be non-negative")

        self.num_prefix_tokens = num_prefix_tokens
        self.residual_scale = residual_scale
        self.state_norm = nn.LayerNorm(d_model)
        self.down = nn.Linear(d_model, bottleneck_dim)
        self.activation = nn.GELU()
        self.up = nn.Linear(bottleneck_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

        if num_prefix_tokens > 0:
            self.prefix_queries = nn.Parameter(torch.randn(num_prefix_tokens, d_model) * 0.02)
            self.prefix_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)
            self.prefix_norm = nn.LayerNorm(d_model)
            self.prefix_gate = nn.Parameter(torch.tensor(float(gate_init)))
        else:
            self.register_parameter("prefix_queries", None)
            self.prefix_attn = None
            self.prefix_norm = None
            self.prefix_gate = None

        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, float(gate_init))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if attention_mask is not None:
            weights = attention_mask.to(dtype=hidden_states.dtype).unsqueeze(-1)
            pooled = (hidden_states * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
            key_padding_mask = attention_mask == 0
        else:
            pooled = hidden_states.mean(dim=1)
            key_padding_mask = None

        residual = self.up(self.dropout(self.activation(self.down(self.state_norm(hidden_states)))))
        gate = torch.sigmoid(self.gate(pooled)).unsqueeze(1)
        adapted = hidden_states + self.residual_scale * gate * residual

        if self.num_prefix_tokens <= 0:
            return adapted, attention_mask

        batch_size = hidden_states.shape[0]
        queries = self.prefix_queries.unsqueeze(0).expand(batch_size, -1, -1)
        prefix, _ = self.prefix_attn(
            queries,
            adapted,
            adapted,
            key_padding_mask=key_padding_mask,
        )
        prefix = torch.sigmoid(self.prefix_gate) * self.prefix_norm(queries + prefix)
        adapted = torch.cat([prefix, adapted], dim=1)

        if attention_mask is None:
            return adapted, None
        prefix_mask = torch.ones(
            batch_size,
            self.num_prefix_tokens,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        return adapted, torch.cat([prefix_mask, attention_mask], dim=1)


class TargetAnchorConditionedResidualAdapter(nn.Module):
    """Anchor-conditioned adapter for low-resource target-domain transfer."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        num_anchors: int,
        num_prefix_tokens: int = 4,
        bottleneck_dim: int = 64,
        dropout: float = 0.1,
        gate_init: float = -4.0,
        residual_scale: float = 1.0,
        anchor_temperature: float = 0.08,
    ) -> None:
        super().__init__()
        if num_anchors <= 0:
            raise ValueError("num_anchors must be positive for TargetAnchorConditionedResidualAdapter")
        if bottleneck_dim <= 0:
            raise ValueError("bottleneck_dim must be positive")
        if num_prefix_tokens < 0:
            raise ValueError("num_prefix_tokens must be non-negative")

        self.num_anchors = num_anchors
        self.num_prefix_tokens = num_prefix_tokens
        self.residual_scale = residual_scale
        self.anchor_temperature = anchor_temperature
        self.register_buffer("anchor_embeddings", torch.zeros(num_anchors, d_model), persistent=True)

        self.state_norm = nn.LayerNorm(d_model)
        self.anchor_query = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )
        self.anchor_film = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 2 * d_model),
        )
        self.down = nn.Linear(d_model, bottleneck_dim)
        self.activation = nn.GELU()
        self.up = nn.Linear(bottleneck_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

        if num_prefix_tokens > 0:
            self.prefix_queries = nn.Parameter(torch.randn(num_prefix_tokens, d_model) * 0.02)
            self.anchor_prefix = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, num_prefix_tokens * d_model),
            )
            self.prefix_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)
            self.prefix_norm = nn.LayerNorm(d_model)
            self.prefix_gate = nn.Parameter(torch.tensor(float(gate_init)))
        else:
            self.register_parameter("prefix_queries", None)
            self.anchor_prefix = None
            self.prefix_attn = None
            self.prefix_norm = None
            self.prefix_gate = None

        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        nn.init.zeros_(self.anchor_film[-1].weight)
        nn.init.zeros_(self.anchor_film[-1].bias)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, float(gate_init))
        if self.anchor_prefix is not None:
            nn.init.zeros_(self.anchor_prefix[-1].weight)
            nn.init.zeros_(self.anchor_prefix[-1].bias)

        self.last_aux_loss: torch.Tensor | None = None
        self.last_anchor_probs: torch.Tensor | None = None

    @torch.no_grad()
    def set_anchor_embeddings(self, anchor_embeddings: torch.Tensor) -> None:
        if anchor_embeddings.shape != self.anchor_embeddings.shape:
            raise ValueError(
                f"Expected anchor embeddings with shape {tuple(self.anchor_embeddings.shape)}, "
                f"got {tuple(anchor_embeddings.shape)}"
            )
        normalized = F.normalize(anchor_embeddings.to(self.anchor_embeddings.device, self.anchor_embeddings.dtype), dim=-1)
        self.anchor_embeddings.copy_(normalized)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        anchor_targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        self.last_aux_loss = None
        self.last_anchor_probs = None

        if attention_mask is not None:
            weights = attention_mask.to(dtype=hidden_states.dtype).unsqueeze(-1)
            pooled = (hidden_states * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
            key_padding_mask = attention_mask == 0
        else:
            pooled = hidden_states.mean(dim=1)
            key_padding_mask = None

        normalized_query = F.normalize(self.anchor_query(pooled), dim=-1)
        normalized_anchors = F.normalize(self.anchor_embeddings.to(hidden_states.dtype), dim=-1)
        anchor_logits = (normalized_query @ normalized_anchors.T) / max(self.anchor_temperature, 1e-6)
        anchor_probs = torch.softmax(anchor_logits, dim=-1)
        anchor_context = anchor_probs @ self.anchor_embeddings.to(hidden_states.dtype)
        self.last_anchor_probs = anchor_probs.detach()

        if anchor_targets is not None and anchor_targets.numel() > 0:
            targets = anchor_targets.to(device=hidden_states.device, dtype=hidden_states.dtype)
            targets = targets / targets.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            self.last_aux_loss = F.kl_div(
                torch.log_softmax(anchor_logits, dim=-1),
                targets,
                reduction="batchmean",
            )

        film = self.anchor_film(anchor_context).unsqueeze(1)
        film_scale, film_shift = film.chunk(2, dim=-1)
        conditioned = self.state_norm(hidden_states) * (1.0 + torch.tanh(film_scale)) + film_shift
        residual = self.up(self.dropout(self.activation(self.down(conditioned))))
        gate_input = torch.cat([pooled, anchor_context], dim=-1)
        gate = torch.sigmoid(self.gate(gate_input)).unsqueeze(1)
        adapted = hidden_states + self.residual_scale * gate * residual

        if self.num_prefix_tokens <= 0:
            return adapted, attention_mask

        batch_size = hidden_states.shape[0]
        queries = self.prefix_queries.unsqueeze(0).expand(batch_size, -1, -1)
        anchor_delta = self.anchor_prefix(anchor_context).view(batch_size, self.num_prefix_tokens, -1)
        queries = queries + anchor_delta
        prefix, _ = self.prefix_attn(
            queries,
            adapted,
            adapted,
            key_padding_mask=key_padding_mask,
        )
        prefix = torch.sigmoid(self.prefix_gate) * self.prefix_norm(queries + prefix)
        adapted = torch.cat([prefix, adapted], dim=1)

        if attention_mask is None:
            return adapted, None
        prefix_mask = torch.ones(
            batch_size,
            self.num_prefix_tokens,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        return adapted, torch.cat([prefix_mask, attention_mask], dim=1)
