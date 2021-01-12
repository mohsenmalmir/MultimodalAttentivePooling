import math
from typing import Optional
import torch
from torch.nn import Module
import torch.nn.functional as F
from torch.nn.common_types import _size_any_t
from torch.nn.modules.conv import _triple

class CorpusAttentivePool(Module):
    kernel_size: _size_any_t
    stride: Optional[_size_any_t]
    padding: _size_any_t
    dilation: _size_any_t
    def __init__(self, kernel_size, stride= 1, padding= 0, dilation= 1, op="max"):
        """

        :param kernel_size: triple, T, H, W
        :param stride: triple, T, H, W
        :param padding: triple, T, H, W
        :param dilation: triple, T, H, W
        :param op: either 'max' or 'mean' over words
        """
        super(CorpusAttentivePool, self).__init__()
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride)
        self.padding = _triple(padding)
        self.dilation = _triple(dilation)
        # aggregating over words
        assert op in ["mean","max"],"op should be either mean or max!"
        self.op = op
        def agg_f(x):
            if op=="max":
                return torch.max(x, dim=1, keepdim=True)[0] # ignore indices
            elif op=="mean":
                return torch.mean(x, dim=1, keepdim=True)
        self.agg = agg_f
        self.k1 = torch.ones(1, 1, *self.kernel_size).float() # required shape is outC,InC,T,H,W


    def forward(self, vw, w):
        """
        perform Pooling based on attention to the language words: w
        :param vw: visual words, the output of convolution layer, NCTHW
        :param w: words, used for attentive pooling, NKC where N is batch size, K is the number of words
        :return:
        """
        # print("inside attentive pool:",vw.shape, w.shape)
        N,C,T,H,W = vw.shape # batch size, inChannels, T, H, W
        # calculate scaled dot product, for each batch use a word filter, then concatenate along the batch
        dotp = []
        for ii in range(N):
            wf = w[ii].unsqueeze(2).unsqueeze(3).unsqueeze(4) # word filter for this batch, of size KC
            dotp.append(self.agg(F.conv3d(vw[ii].unsqueeze(0), wf))) # dot product, NKTHW -> agg -> N1THW
        dotp = torch.cat(dotp,dim=0)
        sdotp = torch.div(dotp, math.sqrt(C)) # scaled dot product, N1THW
        emap = torch.exp(sdotp) # e(x), N1THW
        # convolve with kernel of 1s to calculate the sum, i.e. denominator of the softmax
        denom = F.conv3d(emap, self.k1, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation)# same THW as input map
        # multiply input by emap, to get the scaled visual words
        scaledvw = torch.mul(vw, emap) # NCTHW
        # convolve with k1 to get an average of neighboring visual words
        scaledvw = scaledvw.unsqueeze(2) # NCTHW -> NC1THW
        results = []
        for ii in range(N):
            # C 1 T H W -> C 1 T H W
            avg = F.conv3d(scaledvw[ii], self.k1, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation)# same THW as input map
            results.append(avg.squeeze(1).unsqueeze(0)) # C1THW -> 1CTHW
        avg = torch.cat(results,dim=0) # NCTHW
        # finally, divide by denom
        pooled = torch.div(avg, denom)
        return pooled
    def extra_repr(self):
        return f'kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding},' \
               f'dilation={self.dilation},op="{self.op}"'
