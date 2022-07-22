import torch.nn.functional as F
import torch.nn as nn
import torch
from tacos.norm import *

NEG_INFINITY = float('-inf')
POS_INFINITY = float('inf')

def get_activation_fn(activation):
    if activation == "gelu":
        return F.gelu
    elif activation == "relu":
        return F.relu
    elif activation == "elu":
        return F.elu
    elif activation == "sigmoid":
        return F.sigmoid
    elif activation == "exp":
        return torch.exp
    elif activation == "leak":
        return F.leaky_relu
    elif activation == "1+elu":
        def f(x):
            return 1 + F.elu(x)
        return f
    elif activation == "silu":
        return F.silu
    elif activation == "relu2":
        def f(x):
            return F.relu(x) ** 2
        return f
    else:
        return lambda x: x

def get_norm(norm_type, embed_dim):
    if norm_type == "rmsnorm":
        return RMSNorm(embed_dim)
    elif norm_type == "gatedrmsnorm":
        return GatedRMSNorm(embed_dim)
    elif norm_type == "simplermsnorm":
        return SimpleRMSNorm(embed_dim)
    elif norm_type == "scalenorm":
        return ScaleNorm(embed_dim)
    else:
        return nn.LayerNorm(embed_dim)
    
def linear_product(q, k, v, causal=False, attn_mask=None, eps=1e-4):
    """_summary_

    Args:
        q (Tensor): ..., N, E1, where N is the sequence length, E1 is the embedding dimension.
        k (Tensor): ..., M, E1, where M is the sequence length, E1 is the embedding dimension.
        v (Tensor): ..., M, E2, where M is the sequence length, E2 is the embedding dimension.

    Returns:
        o (Tensor): ..., N, E2
    """
    if causal:
        # to do: fast causal linear product
        n = q.shape[-2]
        m = k.shape[-2]
        if (attn_mask == None):
            attn_mask = (torch.triu(torch.ones(n, m))==1).transpose(0, 1)
            attn_mask = attn_mask.float().masked_fill(attn_mask==0, float('-inf')).to(q)
        weights = torch.einsum('...nd,...md->...nm', q, k)
        weights = weights.masked_fill(attn_mask==float("-inf"), 0)
        denom = weights.sum(dim=-1, keepdim=True)
        denom = torch.clamp_min(denom, eps)
        weights = weights / denom
        output = torch.einsum('...nm,...md->...nd', weights, v)
    else:
        kv = torch.einsum('...nd,...ne->...de', k, v)
        output = torch.einsum('...nd,...de->...ne', q, kv)
        # q(k^T1)
        denom = torch.einsum('...nd,...d->...n', q, torch.sum(k, axis=-2)).unsqueeze(-1)
        denom = torch.clamp_min(denom, eps)
        output = output / denom
        
    return output