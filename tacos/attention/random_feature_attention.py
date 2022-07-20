import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from einops import rearrange

from tacos.utils import linear_product
from tacos.utils import get_activation_fn
from tacos.utils import orthogonal_random_matrix

class RandomFeatureAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        proj_dim=64,
        causal=False,
        dropout=0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.causal = causal
        self.head_dim = self.embed_dim // self.num_heads
        self.proj_dim = proj_dim
        assert (
            self.embed_dim % self.num_heads == 0
        ), "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.proj_matrix = self.build_projection()
        self.dropout = nn.Dropout(dropout)
        
    def build_projection(self):
        w = orthogonal_random_matrix(self.proj_dim, self.head_dim)
        
        return nn.Parameter(w, requires_grad=False)
    
    def normalize(self, x, eps=1e-4):
        norm = torch.norm(x, dim=-1, keepdim=True)
        y = x / (norm + eps)
        
        return y
    
    def projection(self, x):
        y = torch.einsum('...d,...ed->...e', x, self.proj_matrix)
        
        return y
    
    def feature_map(self, x):
        # input shape: ..., n, d
        scale = x.shape[-1] ** -0.5
        feature = torch.cat([torch.sin(x), torch.cos(x)], dim=-1) * scale
        
        return feature

    def forward(
        self,
        x,
        y=None,
        attn_mask=None,
    ):
        """
        - x: (B, N, E), where N is the sequence lenground_truthh, B is the batch size, E is
          the embedding dimension.
        - y: (B, M, E), where M is the sequence lenground_truthh, B is the batch size, E is
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

        # normalize
        q = self.normalize(q)
        k = self.normalize(k)
        
        # projection and scale
        scale = self.num_heads ** 0.25
        q_proj = self.projection(q) * scale
        k_proj = self.projection(k) * scale
        
        # feature map
        q = self.feature_map(q_proj)
        k = self.feature_map(k_proj)

        attn_output = linear_product(q, k, v, causal=self.causal, attn_mask=attn_mask)
        # reshape
        attn_output = rearrange(attn_output, 'b h n d -> b n (h d)')
        # dropout 
        attn_output = self.dropout(attn_output)
        # B, N, E
        attn_output = self.out_proj(attn_output)

        return attn_output
    
    def test(self, x):
        # B, N, E
        q = self.q_proj(x)
        # B, M, E
        k = self.k_proj(x)
        # B, M, D
        v = self.v_proj(x)
        # B, H, N, E
        q, k, v = map(lambda x: rearrange(x, 'b n (h d) -> b h n d', h=self.num_heads), [q, k, v])

        # normalize
        q = self.normalize(q)
        k = self.normalize(k)
        
        # ground truth
        ground_truth = torch.einsum("...nd,...md->...nm", q, k) * (self.num_heads ** -0.5)
        ground_truth = torch.exp(ground_truth)
        
        # projection and scale
        scale = self.num_heads ** 0.25
        q_proj = self.projection(q) * scale
        k_proj = self.projection(k) * scale

        # feature map
        q = self.feature_map(q_proj)
        k = self.feature_map(k_proj)

        qk = torch.einsum("...nd,...md->...nm", q, k)

        return torch.norm(ground_truth - qk) / torch.norm(ground_truth)