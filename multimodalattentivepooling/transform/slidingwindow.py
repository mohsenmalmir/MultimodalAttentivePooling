import torch
import torch.nn.functional as F

class ChunkIoU:
    """
    given a sliding window to a video clip, this class calculates the iou for each sliding window chunk.
    It then thresholds the iou to calculate the classification target
    """
    def __init__(self, window_size, num_chunks, iou, in_name, len_name, dur_name, ts_name, out_name):
        """
        initialize the ChunkIoU
        :param window_size(int): parameter from the network
        :param num_chunks(int): parameter from the network
        :param iou(float): threshold to use for tgt calculation
        :param in_name(str): name of the video field in data
        :param len_name(str): name of the video length in data (in cips)
        :param dur_name(str): name of the duration of the video in seconds
        :param ts_name(str): name of the timestamp of the target
        :param out_name(str): name of the tgt to store in data
        """
        self.window_size = window_size
        self.num_chunks = num_chunks
        self.iou_threshold = iou
        self.in_name = in_name
        self.len_name = len_name
        self.dur_name = dur_name
        self.ts_name = ts_name
        self.out_name = out_name

    def __call__(self, data):
        """
        given a single data batch, this function will add the corresponding tgt for iou threshold for each chunk.
        :param data (dict): containing data from the training loop
        :return: updated data
        """
        # print(len(data))
        clips_in_chunk = self.window_size // self.num_chunks # number of clips in each chunk
        vis_feats = data[self.in_name] # B T C
        B, T, C = vis_feats.shape
        # mark indices as -1 if they are out of the time limit of the video
        indices = torch.arange(T).unsqueeze(0).repeat(B,1)
        indices = indices.unsqueeze(2).unsqueeze(1).float() # B 1 T 1
        # unfold visual feats
        ks = (self.window_size, 1)
        st, pd, dl = (1, 1), (0, 0), (1, 1)
        indices_unfolded = F.unfold(indices, ks, dl, pd, st) # B KS NW
        _, _, NW = indices_unfolded.shape
        indices_unfolded = indices_unfolded.transpose(1,2).view(B, NW, self.num_chunks, clips_in_chunk) # B NW NC CS
        # theoretically you can use index 0 in each chunk to take min, but this is just to keep it general
        starts = torch.min(indices_unfolded,dim=-1)[0]
        ends = torch.max(indices_unfolded,dim=-1)[0]+1 # this is without considering the video length
        # time unit for each clip
        time_unit = 1. / torch.tensor(data[self.len_name])
        time_unit = time_unit.unsqueeze(1).unsqueeze(2).float().repeat(1,NW,self.num_chunks)
        # calculate starts and ends in video length fractions
        starts = starts * time_unit
        ends = ends * time_unit
        # fix for padding
        starts = torch.minimum(starts, torch.tensor(1.))
        ends = torch.minimum(ends, torch.tensor(1.))
        lens = ends - starts
        lens = torch.where(lens>0,lens,torch.tensor(-1.))
        # ground truth intervals
        gt_starts = [d[0]/l for d,l in zip(data[self.ts_name],data[self.dur_name])]
        gt_starts = torch.tensor(gt_starts).unsqueeze(1).unsqueeze(2).repeat(1,NW,self.num_chunks)
        gt_ends = [d[1]/l for d,l in zip(data[self.ts_name],data[self.dur_name])]
        gt_ends = torch.tensor(gt_ends).unsqueeze(1).unsqueeze(2).repeat(1,NW,self.num_chunks)
        gt_lens = gt_ends - gt_starts
        # calculate iou
        intersection = torch.minimum(gt_ends, ends) - torch.maximum(gt_starts, starts)
        union = torch.maximum(ends, gt_ends) - torch.minimum(starts, gt_starts)
        iou = intersection / union
        tgt = torch.where(iou>=self.iou_threshold,torch.tensor(1.),torch.tensor(0.))
        # make sure tgt in padded areas is 0
        tgt = torch.where(starts<1,tgt,torch.tensor(-100.))
        # save to output
        data[self.out_name] = tgt.long()
        return data
