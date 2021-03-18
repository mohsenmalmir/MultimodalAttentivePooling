import torch
from skimage.morphology import binary_opening
import numpy as np

class SegmentTarget:
    """
    This class calculates the target for max-pooled start/end prediction.
    The maxpooling operation maps a sequence of variable length to a fixed-size sequence.
    The start/end targets are calculated by mapping the corresponding timestamp in the duration to the vec size.
    """
    def __init__(self, maxpool_sizes, len_name, dur_name, ts_name, out_names):
        self.maxpool_sizes = maxpool_sizes
        self.len_name = len_name
        self.dur_name = dur_name
        self.ts_name = ts_name
        self.out_names = out_names

    def __call__(self, data):
        starts, ends = zip(*data[self.ts_name])
        starts, ends = torch.tensor(starts), torch.tensor(ends)
        dur = torch.tensor(data[self.dur_name])
        for ii in range(len(self.maxpool_sizes)):
            starts = starts / dur * self.maxpool_sizes[ii]
            ends = ends / dur * self.maxpool_sizes[ii]
            starts = torch.minimum(starts,torch.tensor(self.maxpool_sizes[ii]-1)).int()
            ends = torch.minimum(ends,torch.tensor(self.maxpool_sizes[ii]-1)).int()
            segment = torch.zeros(starts.shape[0],self.maxpool_sizes[ii]).long()
            for jj in range(starts.shape[0]):
                segment[jj, starts[jj]:ends[jj]] = 1
            data[self.out_names[ii]] = segment
        return data

class SegmentTarget2:
    """
    This class calculates the target for max-pooled start/end prediction.
    The start/end targets are calculated by mapping the corresponding timestamp in the duration to the vec size.
    """
    def __init__(self, len_name, dur_name, ts_name, out_name):
        self.len_name = len_name
        self.dur_name = dur_name
        self.ts_name = ts_name
        self.out_name = out_name

    def __call__(self, data):
        starts, ends = zip(*data[self.ts_name])
        starts, ends = torch.tensor(starts), torch.tensor(ends)
        dur = torch.tensor(data[self.dur_name])
        B, L = len(data[self.len_name]), np.max(data[self.len_name])
        segment = torch.zeros([B,L]).long()
        for ii,l in enumerate(data[self.len_name]):
            S = int(starts[ii] / dur[ii] * l)
            E = int(ends[ii] / dur[ii] * l)
            segment[ii,l:] = -100
            segment[ii,S:E+1] = 1
        data[self.out_name] = segment
        # print(segment, self.out_name)
        return data

class ShiftLen:
    """
    """
    def __init__(self, len_name, shift_vals, out_names):
        self.len_name = len_name
        self.shift_vals = shift_vals
        self.out_names = out_names

    def __call__(self, data):
        for s, n in zip(self.shift_vals, self.out_names):
            data[n] = [l-s for l in data[self.len_name]]
            # print(n, data[n])
        return data


class Predict:
    """
    predict the moment from network's segment output.
    """
    def __init__(self, pred_name, len_name, out_name):
        self.pred_name = pred_name
        self.len_name = len_name
        self.out_name = out_name

    def __call__(self, data):
        pred = data[self.pred_name]
        pred = torch.argmax(pred,dim=1)
        # crop based on video length
        pred = [np.asarray(pred[bb,:l].tolist()) for bb,l in zip(range(pred.shape[0]),data[self.len_name])]
        # print(pred[0])
        # suppress noisy predictions
        pred = [binary_opening(p).astype(int) for p in pred]
        # print(pred[0])
        pred = [np.where(p>0)[0] for p in pred]
        pred = [(p[0],p[-1]) if p.shape[0]>0 else (-1,-1) for p in pred]
        # print(pred)
        data[self.out_name] = pred
        return data

    def conclude(self):
        pass