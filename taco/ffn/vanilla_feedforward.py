import torch.nn as nn
from utils import get_act_fun

class VanillaFeedForward(nn.Module):
    def __init__(
        self,
        embed_dim,
        hidden_dim,
        dropout=0.0,
        act_fun="gelu",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act_fun = get_act_fun(act_fun)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)


    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fun(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)

        return x
