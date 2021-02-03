import torch

class MaxPooledStartEnd:
    """
    This class calculates the target for max-pooled start/end prediction.
    The maxpooling operation maps a sequence of variable length to a fixed-size sequence.
    The start/end targets are calculated by mapping the corresponding timestamp in the duration to the vec size.
    """
    def __init__(self, maxpool_size, len_name, dur_name, ts_name, start_outname, end_outname):
        self.maxpool_size = maxpool_size
        self.len_name = len_name
        self.dur_name = dur_name
        self.ts_name = ts_name
        self.start_outname = start_outname
        self.end_outname = end_outname

    def __call__(self, data):
        starts, ends = zip(*data[self.ts_name])
        starts, ends = torch.tensor(starts), torch.tensor(ends)
        dur = torch.tensor(data[self.dur_name])
        starts = starts / dur * self.maxpool_size
        ends = ends / dur * self.maxpool_size
        starts = torch.minimum(starts,torch.tensor(self.maxpool_size-1))
        ends = torch.minimum(ends,torch.tensor(self.maxpool_size-1))
        data[self.start_outname] = starts.long().view(-1,1)
        data[self.end_outname] = ends.long().view(-1,1)
        return data