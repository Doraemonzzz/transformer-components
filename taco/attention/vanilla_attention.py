from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import MINUS_INFINITY

class VanillaAttention(nn.Module):
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
        query, 
        key, 
        value, 
        attn_mask=None
    ):
        """Forward function of vanilla attention.
           B is the batch size, 
           N is the target sequence length, 
           M is the source sequence length, 
           E is the embedding dimension.

        Args:
            query (tensor): (B, N, E)
            key (tensor): (B, M, E)
            value (tensor): (B, M, E)
            attn_mask (tensor): (N, M)

        Returns:
            output (tensor): (B, N, E)
        """
        b, n, e = query.shape
        m = key.shape[1]
        # projection
        # b, n, e; b, m, e
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        # reshape
        # b, h, n, d; b, h, m, d
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), [q, k, v])
        # compute
        # b, h, n, m
        dots = torch.einsum('...nd,...md->...nm', q, k) * self.scale
        # weight
        weight = F.softmax(dots, dim=-1)
        weight = self.dropout(weight)
        if self.causal:
            if attn_mask == None:
                attn_mask = torch.tril(torch.ones(n, n)==1, diagonal=-1).to(q)
            weight = weight.masked_fill(attn_mask==0, MINUS_INFINITY)
        # output
        # b, h, n, d
        output = torch.einsum('...nm,...md->...nd', weight, v)
        # b, n, e
        output = rearrange(output, 'b h n d -> b n (h d)')
        output = self.out_proj(output)

        return output

