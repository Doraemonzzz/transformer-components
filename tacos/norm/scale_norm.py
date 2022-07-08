import torch
from torch import nn

class ScaleNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.d = d
        self.scala = nn.Parameter(torch.ones(1))

    def forward(self, x):
        norm_x = x.norm(2, dim=-1, keepdim=True)
        d_x = self.d
        rms_x = norm_x * d_x ** (-1. / 2)
        x = x * self.scala / (rms_x + self.eps)

        return x