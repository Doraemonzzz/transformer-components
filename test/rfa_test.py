import torch
from tacos.attention import RandomFeatureAttention

b = 10
n = 128
h = 8
e = 256
model = RandomFeatureAttention(e, h)
x = torch.randn(b, n, e)
res = model.test(x)
print(res)