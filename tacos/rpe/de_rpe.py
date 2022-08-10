import torch.nn as nn
import torch

class DeRpe(nn.Module):
    """
        Decomposable relative position encoding base classes
    """
    def __init__(self):
        super().__init__()
        
    def forward(x):
        return x