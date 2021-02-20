import torch

class MaxPooledStartEnd:
    """
    This class calculates the target for max-pooled start/end prediction.
    The maxpooling operation maps a sequence of variable length to a fixed-size sequence.
    The start/end targets are calculated by mapping the corresponding timestamp in the duration to the vec size.
    """
    def __init__(self, maxpool_sizes, len_name, dur_name, ts_name, start_outnames, end_outnames):
        self.maxpool_sizes = maxpool_sizes
        self.len_name = len_name
        self.dur_name = dur_name
        self.ts_name = ts_name
        self.start_outnames = start_outnames
        self.end_outnames = end_outnames

    def __call__(self, data):
        starts, ends = zip(*data[self.ts_name])
        starts, ends = torch.tensor(starts), torch.tensor(ends)
        dur = torch.tensor(data[self.dur_name])
        for ii in range(len(self.maxpool_sizes)):
            starts = starts / dur * self.maxpool_sizes[ii]
            ends = ends / dur * self.maxpool_sizes[ii]
            starts = torch.minimum(starts,torch.tensor(self.maxpool_sizes[ii]-1))
            ends = torch.minimum(ends,torch.tensor(self.maxpool_sizes[ii]-1))
            data[self.start_outnames[ii]] = starts.long().view(-1,1)
            data[self.end_outnames[ii]] = ends.long().view(-1,1)
        return data