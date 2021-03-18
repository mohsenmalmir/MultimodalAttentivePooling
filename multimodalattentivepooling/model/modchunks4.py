from torch.nn import Module, AdaptiveMaxPool1d, Linear,ReLU,ModuleList,ModuleDict,Sequential,Dropout,MultiheadAttention
from torch.nn import LogSoftmax
import torch.nn.functional as F
from multimodalattentivepooling.model.encoder import SeqEncoder
from multimodalattentivepooling.model.sequtils import PositionalEncoding
import torch
import numpy as np


class ModulatedChunks(Module):
    """
    This module implements the modulated chunks idea. Input is assumed to be a set of clips.
    The output is a set of modulated chunks in a sliding window fashion.
    """
    window_size: int
    num_chunks: int
    seq_enc1: SeqEncoder
    seq_enc2: SeqEncoder
    def __init__(self, window_sizes, num_chunks, vis_dim, q_dim, vis_name, len_name, seg_names, lpred_names, rpred_names):
        super(ModulatedChunks, self).__init__()
        # positional encodings of the sequences
        self.vid_pe = PositionalEncoding(vis_dim)
        self.seq_pe = PositionalEncoding(q_dim)
        # window sizes etc.
        self.window_sizes = window_sizes
        self.num_chunks = num_chunks
        self.device = None
        self.vis_dim, self.q_dim = vis_dim, q_dim
        self.seg_names, self.lpred_names, self.rpred_names = seg_names, lpred_names, rpred_names
        self.len_name = len_name
        self.vis_name = vis_name
        # encoders
        self.vid_enc1 = torch.nn.Sequential(Linear(vis_dim, vis_dim),ReLU(),Linear(vis_dim, vis_dim))
        self.vid_enc2 = torch.nn.Sequential(Linear(vis_dim, vis_dim),ReLU(),Linear(vis_dim, vis_dim))
        self.seq_enc1 = torch.nn.Sequential(Linear(q_dim, q_dim),ReLU(),Linear(q_dim, vis_dim))
        self.seq_enc2 = torch.nn.Sequential(Linear(q_dim, q_dim),ReLU(),Linear(q_dim, vis_dim))
        self.seg_pred = Linear(vis_dim, 2)
        self.out = LogSoftmax(dim=1)
        self.lpred = Linear(vis_dim,1)
        self.rpred = Linear(vis_dim,1)


    def to(self, device):
        self.device = device
        super(ModulatedChunks, self).to(device)

    def forward(self,data: dict):
        # video: expected shape of BTC
        vis_feats = data[self.vis_name] # B T C
        vis_feats = self.vid_pe(vis_feats) # positional signal included in the features
        vis_feats = self.vid_enc1(vis_feats)
        B, T, C = vis_feats.shape
        # encode query
        query = data["query_feats"] # BxLxC
        query = self.seq_pe(query) # add positional signals to the query
        enc1 = self.seq_enc1(query) # module labeld '1' in the slide
        enc2 = self.seq_enc2(query) # module labeled '2' in the slide
        clip_word_sim = torch.matmul(vis_feats,enc1.transpose(1,2)) # NC x NWORDS
        _,NUMCLIPS,NUMWORDS = clip_word_sim.shape
        # transpose C to dim=1 to apply unfold
        vis_feats = vis_feats.transpose(1, 2).unsqueeze(3)# convert to [B, C, NC, 1)
        # print(vis_feats.shape)
        st, pd, dl = (1, 1), (0, 0), (1, 1)
        vis_unfolded = [F.unfold(vis_feats, (ws, 1), dl, pd, st) for ws in self.window_sizes]
        vis_unfolded = [v.view(B, C, ws, -1).transpose(1,3) for v,ws in zip(vis_unfolded,self.window_sizes)]
        vis_unfolded = [v.view(B,v.shape[1],self.num_chunks[jj],-1,C) for jj,v in enumerate(vis_unfolded)]
        # vis_unfolded = [v.view(B,v.shape[1],self.num_chunks[jj],-1,C) for jj,v in enumerate(vis_unfolded)]
        # TODO: why using adaptive avg pool, does that mean each 3 clips get merged into a single chunk?
        vis_unfolded = [F.adaptive_avg_pool3d(v, (self.num_chunks[jj],1,C)).squeeze(3) for jj,v in enumerate(vis_unfolded)]
        beliefs_unfolded = [F.unfold(clip_word_sim.transpose(1,2).unsqueeze(3), (ws, 1), dl, pd, st).view(B, NUMWORDS, ws, -1).transpose(1,3) for ws in self.window_sizes]
        beliefs_unfolded = [b.view(b.shape[0],b.shape[1],self.num_chunks[jj],self.window_sizes[jj]//self.num_chunks[jj],b.shape[3]) for jj,b in enumerate(beliefs_unfolded)]
        chunk_labels = [torch.argmax(torch.sum(b,dim=3),dim=3) for b in beliefs_unfolded]
        sel_words = [torch.gather(enc2.unsqueeze(1).repeat(1,l.shape[1],1,1),2,l.unsqueeze(3).repeat(1,1,1,self.vis_dim)) for l in chunk_labels]
        modulated = [sw*vu for sw,vu in zip(sel_words, vis_unfolded)]
        modulated = [m.squeeze() for m in modulated]
        # print([m.shape for m in modulated])
        segments = [self.seg_pred(m).transpose(1,2) for m in modulated]
        # print([s.shape for s in segments])
        lpred = [self.lpred(m).squeeze() for m in modulated]
        # print([l.shape for l in lpred])
        rpred = [self.rpred(m).squeeze() for m in modulated]
        # print([r.shape for r in rpred])
        for ii in range(len(self.seg_names)):
            data[self.seg_names[ii]] = segments[ii]
            data[self.lpred_names[ii]] = lpred[ii]
            data[self.rpred_names[ii]] = rpred[ii]
        return data