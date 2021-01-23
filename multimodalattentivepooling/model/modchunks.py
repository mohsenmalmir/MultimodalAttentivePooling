from typing import Optional
from torch import Tensor
from torch.nn import Module, AdaptiveMaxPool1d, Linear
import torch.nn.functional as F
from torch.nn.modules.transformer import TransformerEncoderLayer, TransformerEncoder, LayerNorm
from .encoder import SeqEncoder
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
    def __init__(self, window_size, num_chunks, seq_enc1, seq_enc2):

        super(ModulatedChunks, self).__init__()
        self.window_size = window_size
        self.num_chunks = num_chunks
        self.seq_enc1 = seq_enc1
        self.seq_enc2 = seq_enc2

    def forward(self,data: dict):
        # video: expected shape of BTC
        vis_feats = data["vis_feats"] # B T C
        B, T, C = vis_feats.shape
        print("input visual features:",vis_feats.shape)
        # encode query
        query = data["query"] # BxLxC
        print("query size:",query.shape)
        enc1 = self.seq_enc1(query) # module labeld '1' in the slide
        print("encoder1:",enc1.shape)
        enc2 = self.seq_enc2(query) # module labeled '2' in the slide
        print("encoder2:",enc2.shape)
        # clip-word similarity
        clip_word_sim = torch.matmul(vis_feats,enc1.transpose(1,2))
        print("word-clip sim:",clip_word_sim.shape)
        clip_labels = torch.argmax(clip_word_sim, dim=2,keepdims=True).unsqueeze(1).float() # Bx1xNCx1
        print("clip labels:",clip_labels.shape)
        # transpose C to dim=1 to apply unfold
        vis_feats = vis_feats.transpose(1, 2).unsqueeze(3)# convert to [B, C, NC, 1)
        print("vis_feats before unfold:",vis_feats.shape)
        # unfold visual feats
        ks = (self.window_size, 1)
        st, pd, dl = (1, 1), (0, 0), (1, 1)
        vis_feats_unfolded = F.unfold(vis_feats, ks, dl, pd, st)
        _, _, NW = vis_feats_unfolded.shape
        vis_feats_unfolded = vis_feats_unfolded.view(B, C, self.window_size, NW).transpose(1,3) # B NW WS C
        print("vis_feats_unfolded:",vis_feats_unfolded.shape)
        # unfold clip labels
        clips_in_chunk = self.window_size // self.num_chunks # number of clips in each chunk
        clip_labels_unfolded = F.unfold(clip_labels, ks, dl, pd, st) # BxWCxNW
        clip_labels_unfolded = clip_labels_unfolded.transpose(1,2).view(B, NW, self.num_chunks, clips_in_chunk)
        print("clip_labels_unfolded:",clip_labels_unfolded.shape)
        # unfold labels to correspond to sliding window
        lbls = clip_labels_unfolded.data.cpu().numpy().astype(int)
        lbls = [[[np.argmax(np.bincount(lbls[b,ii,jj,:])) for jj in range(self.num_chunks)] for ii in range(NW)] for b in range(B)]
        lbls = torch.tensor(lbls).long().unsqueeze(3).repeat(1,1,1,C)
        print("majority vote labels:",lbls.shape)
        # pooling each window to fixed size
        vis_feats_unfolded = vis_feats_unfolded.view(B, NW, self.num_chunks, clips_in_chunk, C)
        print("before pool size:",vis_feats_unfolded.shape)
        pooled = F.adaptive_avg_pool3d(vis_feats_unfolded, (self.num_chunks,1,C)).squeeze(3)# output is B NW NCHUNK C
        print("pooled:",pooled.shape)
        # modulate each chunk
        enc2 = enc2.unsqueeze(1).repeat(1,NW,1,1)
        modulated = torch.gather(enc2,2,lbls)
        print("modulated:",modulated.shape)
        modulated = modulated * pooled
        print("output shape:",modulated.shape)
        return modulated