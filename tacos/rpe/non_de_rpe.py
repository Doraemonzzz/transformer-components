import torch.nn as nn
import torch

class NoDeRpe(nn.Module):
    """
        Non-decomposable relative position encoding base classes
    """
    def __init__(self):
        super().__init__()
        
    def forward(x):
        return x