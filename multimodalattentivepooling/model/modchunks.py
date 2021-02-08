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
    def __init__(self, window_sizes, num_chunks, vis_dim, q_dim, len_name, stend_mxpoolsz, out_names,
                 stpred_name, endpred_name):
        super(ModulatedChunks, self).__init__()
        # positional encodings of the sequences
        self.vid_pe = PositionalEncoding(vis_dim)
        self.seq_pe = PositionalEncoding(q_dim)
        # window sizes etc.
        self.window_sizes = window_sizes
        self.num_chunks = num_chunks
        self.vid_enc1 = torch.nn.Sequential(Linear(vis_dim, vis_dim))#,ReLU(),Linear(vis_dim, vis_dim))
        self.vid_enc2 = torch.nn.Sequential(Linear(vis_dim, vis_dim))#,ReLU(),Linear(vis_dim, vis_dim))
        self.seq_enc1 = torch.nn.Sequential(Linear(q_dim, vis_dim))#,ReLU(),Linear(q_dim, vis_dim))
        self.seq_enc2 = torch.nn.Sequential(Linear(q_dim, vis_dim))#,ReLU(),Linear(q_dim, vis_dim))
        # self.seq_enc1 = Linear(q_dim, vis_dim)
        # self.seq_enc2 = Linear(q_dim, vis_dim)
        # self.seq_enc2 = SeqEncoder(d_model=q_dim, d_out=vis_dim)
        self.pred = dict()
        for jj,name in enumerate(out_names):
            self.pred[name] = Sequential(Linear(self.num_chunks[jj]*vis_dim, vis_dim), # use vis_dim if classifying only chunks, vis_dim*num_chunks for classifying windows
                                         ReLU(),
                                         Dropout(),
                                         Linear(vis_dim, 2)
                                         )
        self.pred = ModuleDict(self.pred)
        self.vis_dim, self.q_dim = vis_dim, q_dim
        # map maxpooled sequence, each of size vis_dim to maxpooled vector
        self.start_pred = Linear(vis_dim*stend_mxpoolsz,stend_mxpoolsz)
        self.end_pred = Linear(vis_dim*stend_mxpoolsz,stend_mxpoolsz)
        self.device = None
        self.len_name = len_name
        self.out_names = out_names
        self.maxpool_startend = AdaptiveMaxPool1d(stend_mxpoolsz)
        self.stpred_name = stpred_name
        self.endpred_name = endpred_name
        # internal counter
        self.iter_counter = 0
        # combine
        self.modulate = MultiheadAttention(1024, 4)

    def to(self, device):
        self.device = device
        super(ModulatedChunks, self).to(device)

    def forward(self,data: dict):
        self.iter_counter += 1 # used for softmax
        temperature = max(1.,-self.iter_counter+25000.) # this will go to 1 at iter counter=25000
        # print(self.iter_counter,temperature)
        # video: expected shape of BTC
        vis_feats = data["vis_feats"] # B T C
        vis_feats = self.vid_pe(vis_feats) # positional signal included in the features
        vis_feats = self.vid_enc1(vis_feats)
        B, T, C = vis_feats.shape
        # print("input visual features:",vis_feats.shape)
        # encode query
        query = data["query_feats"] # BxLxC
        query = self.seq_pe(query) # add positional signals to the query
        # print("query size:",query.shape)
        epsilon = 1.e-7
        enc1 = self.seq_enc1(query) # module labeld '1' in the slide
        # enc1_sum = torch.sum(enc1,dim=2,keepdims=True) + epsilon
        # enc1 = enc1 / enc1_sum # make sure size of the encodings is normed
        # print(enc1)
        # print("encoder1:",enc1.shape)
        enc2 = self.seq_enc2(query) # module labeled '2' in the slide
        # enc2_sum = torch.sum(enc2,dim=2,keepdims=True) + epsilon
        # enc2 = enc2 / enc2_sum # make sure size of the encodings is normed
        # print(enc2)
        # print(enc2)
        # print("encoder2:",enc2.shape)
        # clip-word similarity
        clip_word_sim = torch.matmul(vis_feats,enc1.transpose(1,2)) # NC x NWORDS
        # clip_word_sim = torch.clamp(clip_word_sim,-2.,2.)
        # print(clip_word_sim)
        # print(clip_word_sim)
        # this is modified on Feb 04 to make the word assignment probabilistic
        # clip_word_sim_np = clip_word_sim.data.cpu().numpy()
        # clip_word_sim_np = np.exp(clip_word_sim_np/temperature)
        # clip_word_sim_np = clip_word_sim_np - np.min(clip_word_sim_np,axis=2,keepdims=True)
        # clip_word_sim_np[np.where(clip_word_sim_np==0)] = epsilon
        # clip_word_sim_np_sum = np.sum(clip_word_sim_np,axis=2,keepdims=True)
        # clip_word_sim_np_sum[np.where(clip_word_sim_np_sum==0)] = 1.
        # clip_word_sim_np = clip_word_sim_np / clip_word_sim_np_sum
        # B, NUMC, NUMQ = clip_word_sim_np.shape
        # print("word-clip sim:",clip_word_sim.shape)
        clip_labels = torch.argmax(clip_word_sim, dim=2,keepdims=True).unsqueeze(1).float() # Bx1xNCx1
        # print("clip labels:",clip_labels.shape)
        # transpose C to dim=1 to apply unfold
        vis_feats = vis_feats.transpose(1, 2).unsqueeze(3)# convert to [B, C, NC, 1)
        # print("vis_feats before unfold:",vis_feats.shape)
        # unfold visual feats
        for jj in range(len(self.window_sizes)):
            # clip_labels = [[np.random.choice(NUMQ,p=clip_word_sim_np[bb,kk,:]) for kk in range(NUMC)] for bb in range(B)]
            # clip_labels = torch.tensor(clip_labels).unsqueeze(1).unsqueeze(3).float()
            ks = (self.window_sizes[jj], 1)
            st, pd, dl = (1, 1), (0, 0), (1, 1)
            vis_feats_unfolded = F.unfold(vis_feats, ks, dl, pd, st)
            _, _, NW = vis_feats_unfolded.shape
            vis_feats_unfolded = vis_feats_unfolded.view(B, C, self.window_sizes[jj], NW).transpose(1,3) # B NW WS C
            # print("vis_feats_unfolded:",vis_feats_unfolded.shape)
            # unfold clip labels
            clips_in_chunk = self.window_sizes[jj] // self.num_chunks[jj] # number of clips in each chunk
            clip_labels_unfolded = F.unfold(clip_labels, ks, dl, pd, st) # BxWCxNW
            clip_labels_unfolded = clip_labels_unfolded.transpose(1,2).view(B, NW, self.num_chunks[jj], clips_in_chunk)
            # print("clip_labels_unfolded:",clip_labels_unfolded.shape)
            # unfold labels to correspond to sliding window
            lbls = clip_labels_unfolded.data.cpu().numpy().astype(int)
            lbls = [[[np.argmax(np.bincount(lbls[b,ii,jj,:])) for jj in range(self.num_chunks[jj])] for ii in range(NW)] for b in range(B)]
            if jj==0 and np.random.uniform(0,1)<.01:
                print(lbls[0])
            # print(lbls)
            lbls = torch.tensor(lbls).long().unsqueeze(3).repeat(1,1,1,C)
            if self.device:
                lbls = lbls.to(self.device)
            # print("majority vote labels:",lbls.shape)
            # pooling each window to fixed size
            vis_feats_unfolded = vis_feats_unfolded.view(B, NW, self.num_chunks[jj], clips_in_chunk, C)
            # print("before pool size:",vis_feats_unfolded.shape)
            pooled = F.adaptive_avg_pool3d(vis_feats_unfolded, (self.num_chunks[jj],1,C)).squeeze(3)# output is B NW NCHUNK C
            pooled = self.vid_enc2(pooled)
            # print("pooled:",pooled.shape)
            # modulate each chunk
            enc2_weights = enc2.unsqueeze(1).repeat(1,NW,1,1)
            # print(enc2_weights[:,0,:,0])
            modulated = torch.gather(enc2_weights,2,lbls)
            # print(modulated[0,0,0,:])
            # x1 = modulated[0,0,0,:].data.cpu().numpy()
            # x2 = np.asarray([enc2_weights[0,0,lbls[0,0,0,ll],ll].item() for ll in range(C)])
            # print((x1==x2).sum())
            pooled = pooled.view(B,NW*self.num_chunks[jj],self.vis_dim).transpose(0,1)
            modulated = modulated.view(B,NW*self.num_chunks[jj],self.vis_dim).transpose(0,1)
            modulated, _ = self.modulate(pooled,modulated,pooled)
            modulated = modulated.transpose(0,1).contiguous().view(B,NW,self.num_chunks[jj],self.vis_dim)
            # modulated = modulated * pooled # B NW NC C
            # print(B,NW,self.vis_dim * self.num_chunks[jj])
            modulated2 = modulated.view(B,NW,self.vis_dim * self.num_chunks[jj])
            # print(modulated2.shape)
            # window classification
            # modulated = modulated.view(B,NW,-1)
            # data[self.out_names[jj]] = self.pred[jj](modulated).transpose(1,2)
            # print(data[self.out_names[jj]].shape)
            # print(modulated.shape)
            # print(modulated)
            # modulated = pooled
            # print("output shape:",modulated.shape)
            # data["pred"] = self.pred(modulated).squeeze(-1)
            # data[self.out_names[jj]] = self.pred(modulated)
            # this is to make sure the output directly works with BCEloss, e.g. B N_Classes D1 D2 ...
            # chunk classification
            # data[self.out_names[jj]] = self.pred[self.out_names[jj]](modulated).transpose(2,3).transpose(1,2)
            data[self.out_names[jj]] = self.pred[self.out_names[jj]](modulated2).transpose(1,2)
            if jj==0:
                _, _, _, D = modulated.shape
                # modulated has shape B NW NC CS
                # sequence length is len - window_sizes[0] + 1 or len???
                modulated = [modulated[bb,:l,:,:].view(-1,D).unsqueeze(0).transpose(1,2) for bb,l in zip(range(B),data[self.len_name])]
                # max-pool
                max_pooled = torch.cat([self.maxpool_startend(m) for m in modulated],dim=0)
                max_pooled = max_pooled.view(B, -1)
                data[self.stpred_name] = self.start_pred(max_pooled).unsqueeze(2)
                data[self.endpred_name] = self.end_pred(max_pooled).unsqueeze(2)
                # break
        return data