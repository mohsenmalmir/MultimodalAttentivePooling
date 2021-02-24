from torch.nn import Module, AdaptiveMaxPool1d, Linear,ReLU,ModuleList,ModuleDict,Sequential,Dropout,MultiheadAttention
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
    def __init__(self, window_sizes, num_chunks, vis_dim, q_dim, model_dim, len_name, stend_mxpoolszs,
                 stpred_names, endpred_names, seg_names):
        super(ModulatedChunks, self).__init__()
        # positional encodings of the sequences
        self.vid_pe = PositionalEncoding(vis_dim)
        self.seq_pe = PositionalEncoding(q_dim)
        # window sizes etc.
        self.window_sizes = window_sizes
        self.num_chunks = num_chunks
        self.vid_enc1 = torch.nn.Sequential(Linear(vis_dim, vis_dim),ReLU(),Linear(vis_dim, vis_dim))
        self.vid_enc2 = torch.nn.Sequential(Linear(vis_dim, vis_dim),ReLU(),Linear(vis_dim, vis_dim))
        self.seq_enc1 = torch.nn.Sequential(Linear(q_dim, q_dim),ReLU(),Linear(q_dim, vis_dim))
        self.seq_enc2 = torch.nn.Sequential(Linear(q_dim, q_dim),ReLU(),Linear(q_dim, vis_dim))
        self.vis_dim, self.q_dim, self.model_dim = vis_dim, q_dim, model_dim
        # map maxpooled sequence, each of size vis_dim to maxpooled vector
        self.model_proj = Linear(vis_dim, model_dim)
        start_pred = dict()
        for st_name,mxpool_sz in zip(stpred_names,stend_mxpoolszs):
            start_pred[st_name] = Linear(model_dim*mxpool_sz,mxpool_sz)
        self.start_pred = ModuleDict(start_pred)
        end_pred = dict()
        for end_name,mxpool_sz in zip(endpred_names,stend_mxpoolszs):
            end_pred[end_name] = Linear(model_dim*mxpool_sz,mxpool_sz)
        self.end_pred = ModuleDict(end_pred)
        seg_pred = dict()
        for seg_name,mxpool_sz in zip(seg_names,stend_mxpoolszs):
            seg_pred[seg_name] = Linear(model_dim,2)
        self.seg_pred = ModuleDict(seg_pred)
        self.device = None
        self.len_name = len_name
        maxpool_startend = dict()
        for ii in range(len(stend_mxpoolszs)):
            maxpool_startend[str(ii)] = AdaptiveMaxPool1d(stend_mxpoolszs[ii])
        self.maxpool_startend = ModuleDict(maxpool_startend)
        self.start_names, self.end_names  = stpred_names, endpred_names
        self.seg_names = seg_names

    def to(self, device):
        self.device = device
        super(ModulatedChunks, self).to(device)

    def forward(self,data: dict):
        # video: expected shape of BTC
        vis_feats = data["vis_feats"] # B T C
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
        st, pd, dl = (1, 1), (0, 0), (1, 1)
        clip_index = torch.arange(T).unsqueeze(0).repeat(B,1).unsqueeze(1).unsqueeze(3).float()
        vis_unfolded = [F.unfold(vis_feats, (ws, 1), dl, pd, st).view(B, C, ws, -1).transpose(1,3) for ws in self.window_sizes]
        vis_unfolded = [v.view(B,v.shape[1],self.num_chunks[jj],-1,C) for jj,v in enumerate(vis_unfolded)]
        vis_unfolded = [F.adaptive_avg_pool3d(v, (self.num_chunks[jj],1,C)).squeeze(3) for jj,v in enumerate(vis_unfolded)]
        beliefs_unfolded = [F.unfold(clip_word_sim.transpose(1,2).unsqueeze(3), (ws, 1), dl, pd, st).view(B, NUMWORDS, ws, -1).transpose(1,3) for ws in self.window_sizes]
        beliefs_unfolded = [b.view(b.shape[0],b.shape[1],self.num_chunks[jj],self.window_sizes[jj]//self.num_chunks[jj],b.shape[3]) for jj,b in enumerate(beliefs_unfolded)]
        chunk_labels = [torch.argmax(torch.sum(b,dim=3),dim=3) for b in beliefs_unfolded]
        sel_words = [torch.gather(enc2.unsqueeze(1).repeat(1,l.shape[1],1,1),2,l.unsqueeze(3).repeat(1,1,1,self.vis_dim)) for l in chunk_labels]
        modulated = [(sw*vu).view(B,vu.shape[1],-1) for sw,vu in zip(sel_words, vis_unfolded)]
        for jj in range(len((self.start_pred.keys()))):
            M = modulated[jj].view(B,modulated[jj].shape[1],self.num_chunks[jj],self.vis_dim)
            M = self.model_proj(M)
            M = [M[bb,:l,:,:].view(-1,self.model_dim).unsqueeze(0).transpose(1,2) for bb,l in zip(range(B),data[self.len_name])]
            # max-pool
            max_pooled = torch.cat([self.maxpool_startend[str(jj)](m) for m in M],dim=0)
            # predict segment
            data[self.seg_names[jj]] = self.seg_pred[self.seg_names[jj]](max_pooled.transpose(1,2)).transpose(1,2)
            # predict start/end
            max_pooled = max_pooled.view(B, -1)
            data[self.start_names[jj]] = self.start_pred[self.start_names[jj]](max_pooled).unsqueeze(2)
            data[self.end_names[jj]] = self.end_pred[self.end_names[jj]](max_pooled).unsqueeze(2)
        return data