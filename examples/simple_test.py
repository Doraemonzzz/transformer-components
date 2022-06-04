import torch
from xformers import VanillaTransformer

num_layers = 2
embed_dim = 128
hidden_dim = 4 * embed_dim
num_heads = 4
causal = True
attn_dropout = 0.0
ffn_dropout = 0.0
attn_type = "vanilla"
ffn_type = "vanilla"
norm_type = "ln"
pre_norm = False
act_fun = "gelu"

model = VanillaTransformer(
    num_layers=num_layers,
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

print(model)

model.cuda()

b = 10
l = 100
e = embed_dim

X = torch.rand(b, l, e).cuda()
Y = model(X)
