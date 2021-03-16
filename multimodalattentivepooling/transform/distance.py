import torch

class NormalizedLRDist:
    """
    This class calculates the distance from left/right of the target moment.
    The distance is normalized to [0,1]
    it is -100 outside the moment
    """
    def __init__(self, vis_name, len_name, dur_name, ts_name, lout_name, rout_name):
        self.vis_name = vis_name
        self.len_name = len_name
        self.dur_name = dur_name
        self.ts_name = ts_name
        self.lout_name = lout_name
        self.rout_name = rout_name

    def __call__(self, data):
        starts, ends = zip(*data[self.ts_name])
        starts, ends = torch.tensor(starts), torch.tensor(ends)
        dur = torch.tensor(data[self.dur_name])
        B, L, _ = data[self.vis_name].shape
        ltgt = torch.empty([B,L]).fill_(-100).float()
        rtgt = torch.empty([B,L]).fill_(-100).float()
        for ii,l in enumerate(data[self.len_name]):
            S = int(starts[ii] / dur[ii] * l)
            E = int(ends[ii] / dur[ii] * l)
            SZ = ltgt[ii,S:E+1].shape[0]
            ltgt[ii,S:E+1] = torch.arange(SZ) / float(SZ)
            rtgt[ii,S:E+1] = torch.arange(SZ) / float(SZ)
        data[self.lout_name] = ltgt
        data[self.rout_name] = rtgt
        return data