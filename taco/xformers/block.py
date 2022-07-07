import torch.nn as nn
import torch.nn.functional as F
from ..attention import VanillaAttention
from ..ffn import VanillaFeedForward
from ..utils import get_norm

class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim,
        hidden_dim,
        num_heads,
        causal=False,
        attn_dropout=0.0,
        ffn_dropout=0.0,
        attn_type="vanilla",
        ffn_type="vanilla",
        norm_type="ln",
        pre_norm=False,
        act_fun="gelu",
    ):
        super().__init__()
        if attn_type == "vanilla":
            self.attn = VanillaAttention(embed_dim=embed_dim, num_heads=num_heads, causal=causal, dropout=attn_dropout)
        else:
            self.attn = VanillaAttention(embed_dim=embed_dim, num_heads=num_heads, causal=causal, dropout=attn_dropout)

        if ffn_type == "vanilla":
            self.ffn = VanillaFeedForward(embed_dim=embed_dim, hidden_dim=hidden_dim, act_dropout=ffn_dropout, act_fun=act_fun)
        else:
            self.ffn = VanillaFeedForward(embed_dim=embed_dim, hidden_dim=hidden_dim, act_dropout=ffn_dropout, act_fun=act_fun)

        self.attn_norm = get_norm(norm_type, embed_dim)
        self.ffn_norm = get_norm(norm_type, embed_dim)

        self.pre_norm = pre_norm
        if self.pre_norm:
            self.forward = self.forward_pre_norm
        else:
            self.forward = self.forward_post_norm

    def forward_pre_norm(self, x, y=None):
        # attention
        x = self.attn_norm(x)
        if y == None:
            y = x
        x = x + self.attn(x, y, y)
        # ffn
        x = self.ffn_norm(x)
        x = x + self.ffn(x)

        return x

    def forward_post_norm(self, x, y=None):
        # attention
        if y == None:
            y = x
        x = x + self.attn(x, y, y)
        x = self.attn_norm(x)
        # ffn
        x = x + self.ffn(x)
        x = self.ffn_norm(x)

        return x