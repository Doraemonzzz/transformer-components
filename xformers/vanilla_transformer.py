import torch.nn as nn
import torch.nn.functional as F
from .block import TransformerBlock

class VanillaTransformer(nn.Module):
    def __init__(
        self,
        num_layers,
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
        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(
                TransformerBlock(
                    embed_dim=embed_dim,
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    causal=causal,
                    attn_dropout=attn_dropout,
                    ffn_dropout=ffn_dropout,
                    attn_type=attn_type,
                    ffn_type=ffn_type,
                    norm_type=norm_type,
                    pre_norm=pre_norm,
                    act_fun=act_fun,
                )
            )

    def forward(self, x, y=None):
        for layer in self.layers:
            x = layer(x)

        return x