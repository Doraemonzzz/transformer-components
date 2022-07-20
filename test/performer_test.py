import torch
from tacos.attention import PerformerAttention

b = 10
n = 128
h = 8
e = 256
model = PerformerAttention(e, h, 64)
x = torch.randn(b, n, e)
res = model.test(x)
print(res)