import torch.nn as nn
import torch

from .de_rpe import DeRpe

class RoPE(DeRpe):
    def __init__(self, rope_dim, theta_type="default", learned=False):
        super().__init__()
        self.e = (rope_dim // 2) * 2
        self.theta_type = theta_type
        self.learned = learned
        if self.learned:
            self.theta = nn.Parameter(self.compute_theta())
    
    def compute_theta(self):
        if self.theta_type == "random":
            theta = torch.rand(self.e // 2)
        else:
            theta = 10000 ** (-2 / self.e * torch.arange(self.e // 2))
            
        return theta
    
    def get_theta(self):
        if self.learned:
            return self.theta
        else:
            return self.compute_theta()
    
    def forward(self, x, dims=[-2]):
        for dim in dims:
            x = self.rope(x, dim)
        return x

    def rope(self, x, dim=-2):
        d = x.shape[-1]
        # do rope in first ([d // 2] * 2) feature
        e = (d // 2) * 2
        assert e == self.e
        
        return self.transform(x, dim)
    
    def transform(self, x, dim=-2):
        """rope transform
        - x: (..., E).
        - dim: the dimension operated on.
        """
        n = len(x.shape)
        d = x.shape[-1]
        e = self.e
        if dim < 0:
            dim = n + dim
        l = x.shape[dim]
        # last several feature(not doing rope)
        x1 = x[..., e:]
        # first e feature(doing rope)
        x2 = x[..., :e]
        
        # compute theta
        # (e // 2, )
        theta = self.get_theta().to(x)
        # (1, e)
        theta = torch.stack([theta, theta], dim=-1).reshape(1, -1)
        # change to the same dimension as x: (..., e)
        for _ in range(n - 2):
            theta = theta.unsqueeze(0)
            
        # compute index
        index = torch.arange(l).to(x)
        # change to the same dimension as x: (..., l, ...)
        for _ in range(dim):
            index = index.unsqueeze(0)
        for _ in range(n - dim - 1):
            index = index.unsqueeze(-1)

        # get theta
        theta = theta * index
        
        # transform
        # (-q1, -q3), (q0, q2) -> (-q1, q0, -q3, q2)
        x3 = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).reshape_as(x)
        x_transform = x2 * torch.cos(theta) + x3 * torch.sin(theta)

        if e != d:
            x_transform = torch.cat([x_transform, x1], dim=-1)

        return x_transform

        x_transform = x * torch.cos(theta) + x_half * torch.sin(theta)

        if e != d:
            x_transform = torch.cat([x_transform, x1], dim=-1)

        return x_transform