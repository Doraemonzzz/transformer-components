import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

class VanillaAttention(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        causal=False,
        dropout=0.0,
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
        self.scale = self.head_dim ** -0.5

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

        b, h, n, d = q.shape
        scaling = d ** -0.5
        score = torch.einsum('bhnd,bhmd->bhnm', q, k) * scaling
        q = q * scaling

        if self.causal:
            if attn_mask == None:
                attn_mask = (torch.triu(torch.ones(n, n))==1).transpose(0, 1)
                attn_mask = attn_mask.float().masked_fill(attn_mask==0, float('-inf')).to(q)
            score = score.masked_fill(attn_mask==float("-inf"), float("-inf"))
        weights = F.softmax(score, dim=-1)
        weights = self.dropout(weights)
        attn_output = torch.einsum('bhnm,bhmd->bhnd', weights, v)
        # reshape
        attn_output = rearrange(attn_output, 'b h n d -> b n (h d)')
        # B, N, E
        attn_output = self.out_proj(attn_output)

        return attn_output