from typing import Optional

import torch
import torch.nn as nn


class SlotAttention(nn.Module):
    """Implementation of SlotAttention.

    Based on the slot attention implementation of Phil Wang available at:
    https://github.com/lucidrains/slot-attention
    """

    def __init__(
            self,
            dim: int,
            feature_dim: int,
            kvq_dim: Optional[int] = None,
            n_heads: int = 1,
            iters: int = 3,
            eps: float = 1e-8,
            ff_mlp: Optional[nn.Module] = None,
            use_projection_bias: bool = False,
            use_implicit_differentiation: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.iters = iters
        self.eps = eps
        self.use_implicit_differentiation = use_implicit_differentiation

        if kvq_dim is None:
            self.kvq_dim = dim
        else:
            self.kvq_dim = kvq_dim

        if self.kvq_dim % self.n_heads != 0:
            raise ValueError("Key, value, query dimensions must be divisible by number of heads.")
        self.dims_per_head = self.kvq_dim // self.n_heads
        self.scale = self.dims_per_head ** -0.5

        self.to_q = nn.Linear(dim, self.kvq_dim, bias=use_projection_bias)
        self.to_k = nn.Linear(feature_dim, self.kvq_dim, bias=use_projection_bias)
        self.to_v = nn.Linear(feature_dim, self.kvq_dim, bias=use_projection_bias)

        self.gru = nn.GRUCell(self.kvq_dim, dim)

        self.norm_input = nn.LayerNorm(feature_dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.ff_mlp = ff_mlp
        self.norm_mlp = nn.LayerNorm(dim)

    def step(self, slots, k, v, masks=None):
        bs, n_slots, _ = slots.shape
        slots_prev = slots

        slots = self.norm_slots(slots)
        q = self.to_q(slots).view(bs, n_slots, self.n_heads, self.dims_per_head)

        dots = torch.einsum("bihd,bjhd->bihj", q, k) * self.scale
        if masks is not None:
            # Masked slots should not take part in the competition for features. By replacing their
            # dot-products with -inf, their attention values will become zero within the softmax.
            dots.masked_fill_(masks.to(torch.bool).view(bs, n_slots, 1, 1), float("-inf"))

        attn = dots.flatten(1, 2).softmax(dim=1)  # Take softmax over slots and heads
        attn = attn.view(bs, n_slots, self.n_heads, -1)
        attn_before_reweighting = attn
        attn = attn + self.eps
        attn = attn / attn.sum(dim=-1, keepdim=True)

        updates = torch.einsum("bjhd,bihj->bihd", v, attn)

        slots = self.gru(updates.reshape(-1, self.kvq_dim), slots_prev.reshape(-1, self.dim))

        slots = slots.reshape(bs, -1, self.dim)


        if self.ff_mlp:
            slots = slots + self.ff_mlp(self.norm_mlp(slots))

        return slots, attn_before_reweighting.mean(dim=2)

    def iterate(self, slots, k, v, masks=None):
        for _ in range(self.iters):
            slots, attn = self.step(slots, k, v, masks)
        return slots, attn

    def forward(
            self, inputs: torch.Tensor, conditioning: torch.Tensor, masks: Optional[torch.Tensor] = None
    ):
        b, n, d = inputs.shape
        slots = conditioning

        inputs = self.norm_input(inputs)
        k = self.to_k(inputs).view(b, n, self.n_heads, self.dims_per_head)
        v = self.to_v(inputs).view(b, n, self.n_heads, self.dims_per_head)

        if self.use_implicit_differentiation:
            slots, attn = self.iterate(slots, k, v, masks)
            slots, attn = self.step(slots.detach(), k, v, masks)
        else:
            slots, attn = self.iterate(slots, k, v, masks)

        return slots, attn
