import torch.nn.functional as F
import torch.nn as nn
import torch

MINUS_INFINITY = 1e-8

def get_act_fun(act_fun):
    if act_fun == "gelu":
        return F.gelu
    elif act_fun == "relu":
        return F.relu
    elif act_fun == "elu":
        return F.elu
    elif act_fun == "sigmoid":
        return F.sigmoid
    elif act_fun == "exp":
        return torch.exp
    elif act_fun == "leak":
        def f(x):
            return F.leaky_relu(x)
        return f
    elif act_fun == "1+elu":
        def f(x):
            return 1 + F.elu(x)
        return f
    elif act_fun == "silu":
        return F.silu
    else:
        return lambda x: x

def get_norm(norm_type, embed_dim):
    if norm_type == "ln":
        return nn.LayerNorm(embed_dim)
    else:
        return nn.LayerNorm(embed_dim)