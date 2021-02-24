import torch

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