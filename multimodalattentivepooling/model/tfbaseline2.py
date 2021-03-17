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
    def __init__(self, vis_dim, q_dim, len_name, qlen_name, segment_name, lpred_name, rpred_name):
        super(Baseline, self).__init__()
        # positional encodings of the sequences
        self.vid_pe = PositionalEncoding(vis_dim)
        self.seq_pe = PositionalEncoding(q_dim)
        self.vid_enc = SeqEncoder(d_model=vis_dim, d_out=vis_dim,num_layers=2)
        self.query_enc = SeqEncoder(d_model=q_dim, d_out=vis_dim,num_layers=2)
        # self.vid_enc = torch.nn.Sequential(Linear(vis_dim, vis_dim),ReLU(),Linear(vis_dim, vis_dim))
        # self.query_enc = torch.nn.Sequential(Linear(q_dim, q_dim),ReLU(),Linear(q_dim, vis_dim))
        # self.fusion = SeqEncoder(d_model=vis_dim, d_out=vis_dim)
        self.vis_dim, self.q_dim = vis_dim, q_dim
        self.seg_pred = Linear(vis_dim, 2)
        self.device = None
        self.len_name = len_name
        self.qlen_name = qlen_name
        self.segment_name = segment_name
        self.mha0 = MultiheadAttention(vis_dim,2)
        self.mha = MultiheadAttention(vis_dim,2)
        self.out = LogSoftmax(dim=1)
        self.lpred_name = lpred_name
        self.rpred_name = rpred_name
        self.lpred = Linear(vis_dim,1)
        self.rpred = Linear(vis_dim,1)


    def to(self, device):
        self.device = device
        super(Baseline, self).to(device)

    def forward(self,data: dict):
        # video: expected shape of BTC
        vis_feats = data["vis_feats"] # B T C
        B,L,_ = vis_feats.shape
        vis_feats = self.vid_pe(vis_feats) # positional signal included in the features
        vis_feats = self.vid_enc(vis_feats)

        # encode query
        query = data["query_feats"] # BxLxC
        query = self.seq_pe(query) # add positional signals to the query
        query = self.query_enc(query) # module labeld '1' in the slide
        # MHA0
        qmha = self.mha0(vis_feats.transpose(0,1),query.transpose(0,1),query.transpose(0,1))[0].transpose(0,1)
        # transpose C to dim=1 to apply unfold
        query = [query[b,:l,:].unsqueeze(0) for b,l in zip(range(B),data[self.qlen_name])]
        query = [AdaptiveMaxPool1d(l)(q.transpose(1,2)).transpose(1,2) for q,l in zip(query,data[self.len_name])]
        query = [F.pad(q,(0,0,0,L-q.shape[1])) for q in query]
        query = torch.cat(query,dim=0)
        vis_feats = self.mha(query.transpose(0,1),vis_feats.transpose(0,1),vis_feats.transpose(0,1))[0].transpose(0,1)
        vis_feats = vis_feats + qmha
        data[self.segment_name] = self.out(self.seg_pred(vis_feats).transpose(1,2))
        data[self.lpred_name] = self.lpred(vis_feats).squeeze(2)
        data[self.rpred_name] = self.rpred(vis_feats).squeeze(2)
        return data