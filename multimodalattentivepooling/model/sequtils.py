import torch
import math
from torch import nn

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=128):
        super(PositionalEncoding, self).__init__()
        # self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)#.transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # make it to TBC
        # x = x.transpose(0,1)
        # x = x + self.pe[x.size(0), :, :]
        x = x + self.pe[:, :x.size(1), :]

        # back to BTC
        # x = x.transpose(0,1)
        return x