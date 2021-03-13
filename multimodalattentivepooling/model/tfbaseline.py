from torch.nn import Module, AdaptiveMaxPool1d, Linear,ReLU,ModuleList,ModuleDict,Sequential,Dropout,MultiheadAttention
from torch.nn import Conv1d, LogSoftmax
import torch.nn.functional as F
from multimodalattentivepooling.model.encoder import SeqEncoder
from multimodalattentivepooling.model.sequtils import PositionalEncoding
import torch
import numpy as np


class Baseline(Module):
    """
    This module implements the modulated chunks idea. Input is assumed to be a set of clips.
    The output is a set of modulated chunks in a sliding window fashion.
    """
    def __init__(self, vis_dim, q_dim, len_name, pred_len, start_name, end_name):
        super(Baseline, self).__init__()
        # positional encodings of the sequences
        self.vid_pe = PositionalEncoding(vis_dim)
        self.seq_pe = PositionalEncoding(q_dim)
        self.vid_enc = SeqEncoder(d_model=vis_dim, d_out=vis_dim)
        self.query_enc = SeqEncoder(d_model=q_dim, d_out=vis_dim)
        # self.vid_enc = torch.nn.Sequential(Linear(vis_dim, vis_dim),ReLU(),Linear(vis_dim, vis_dim))
        # self.query_enc = torch.nn.Sequential(Linear(q_dim, q_dim),ReLU(),Linear(q_dim, vis_dim))
        self.vis_dim, self.q_dim = vis_dim, q_dim
        self.start_pred = Linear(vis_dim, 1)
        self.end_pred = Linear(vis_dim, 1)
        # self.start_pred = Conv1d(vis_dim, 1, 21, padding=10)
        # self.end_pred = Conv1d(vis_dim, 1, 21, padding=10)
        self.vid_pool = AdaptiveMaxPool1d(pred_len)
        self.q_pool = AdaptiveMaxPool1d(pred_len)
        self.device = None
        self.len_name = len_name
        self.start_name, self.end_name = start_name, end_name
        self.mha = MultiheadAttention(vis_dim,2)
        self.out = LogSoftmax(dim=1)


    def to(self, device):
        self.device = device
        super(Baseline, self).to(device)

    def forward(self,data: dict):
        # video: expected shape of BTC
        vis_feats = data["vis_feats"] # B T C
        vis_feats = self.vid_pe(vis_feats) # positional signal included in the features
        vis_feats = self.vid_enc(vis_feats)
        vis_feats = self.vid_pool(vis_feats.transpose(1,2)).transpose(1,2)
        # encode query
        query = data["query_feats"] # BxLxC
        query = self.seq_pe(query) # add positional signals to the query
        query = self.query_enc(query) # module labeld '1' in the slide
        query = self.q_pool(query.transpose(1,2)).transpose(1,2)
        # transpose C to dim=1 to apply unfold
        vis_feats = self.mha(query.transpose(0,1),vis_feats.transpose(0,1),vis_feats.transpose(0,1))[0].transpose(0,1)
        vis_feats = vis_feats + query
        # print(vis_feats.shape)
        # vis_feats = vis_feats.contiguous().view(vis_feats.shape[0],-1)
        # data[self.start_name] = self.start_pred(vis_feats.transpose(1,2)).transpose(1,2)
        # data[self.end_name] = self.end_pred(vis_feats.transpose(1,2)).transpose(1,2)
        data[self.start_name] = self.out(self.start_pred(vis_feats).squeeze(1))
        data[self.end_name] = self.out(self.end_pred(vis_feats).squeeze(1))
        return data