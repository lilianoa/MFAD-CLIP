import torch
import torch.nn as nn
from torch.nn import functional as F
from collections import OrderedDict

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class cross_attn(torch.nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super(cross_attn, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        # self.norm1 = nn.Sequential(
        #     nn.BatchNorm1d(d_model),
        #     nn.ReLU(),
        # )

    def forward(self, x1, x2):
        x = self.multihead_attn(query=x1, key=x2, value=x2)[0]
        x1 = x1 + self.norm1(x)
        return x1

class IT_fusion(torch.nn.Module):
    def __init__(self, input_dim, nhead, dropout):
        super(IT_fusion, self).__init__()
        self.TtoI_attn = cross_attn(input_dim, nhead, dropout=dropout)
        self.ItoT_attn = cross_attn(input_dim, nhead, dropout=dropout)

    def forward(self, text, image):
        x = self.TtoI_attn(text, image)
        x = self.ItoT_attn(image, x)

        return x