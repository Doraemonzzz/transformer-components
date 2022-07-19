import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from ..utils import linear_product
from ..utils import get_activation_fn

class LinearAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        causal=False,
        dropout=0.0,
        act_fun="1+elu",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.causal = causal
        self.head_dim = self.embed_dim // self.num_heads
        assert (
            self.embed_dim % self.num_heads == 0
        ), "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = get_activation_fn(act_fun)

    def forward(
        self,
        x,
        y=None,
        attn_mask=None,
    ):
        """
        - x: (B, N, E), where N is the sequence length, B is the batch size, E is
          the embedding dimension.
        - y: (B, M, E), where M is the sequence length, B is the batch size, E is
          the embedding dimension.
        - attn_mask: (N, M).
        """
        if y == None:
            y = x
        # B, N, E
        q = self.q_proj(x)
        # B, M, E
        k = self.k_proj(y)
        # B, M, D
        v = self.v_proj(y)
        # B, H, N, E
        q, k, v = map(lambda x: rearrange(x, 'b n (h d) -> b h n d', h=self.num_heads), [q, k, v])

        # act
        q = self.act(q)
        k = self.act(k)

        attn_output = linear_product(q, k, v, causal=self.causal, attn_mask=attn_mask)
        # reshape
        attn_output = rearrange(attn_output, 'b h n d -> b n (h d)')
        # dropout 
        attn_output = self.dropout(attn_output)
        # B, N, E
        attn_output = self.out_proj(attn_output)

        return attn_output