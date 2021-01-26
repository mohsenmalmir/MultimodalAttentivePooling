import torch
import torch.nn.functional as F

class ChunkIoU:
    """
    given a sliding window to a video clip, this class calculates the iou for each sliding window chunk.
    It then thresholds the iou to calculate the classification target
    """
    def __init__(self, window_size, num_chunks, iou, in_name, out_name):
        """
        initialize the ChunkIoU
        :param window_size(int): parameter from the network
        :param num_chunks(int): parameter from the network
        :param iou(float): threshold to use for tgt calculation
        :param in_name(str): name of the video field in data
        :param out_name(str): name of the tgt to store in data
        """
        self.window_size = window_size
        self.num_chunks = num_chunks
        self.iou_threshold = iou
        self.in_name = in_name
        self.out_name = out_name

    def __call__(self, data):
        """
        given a single data batch, this function will add the corresponding tgt for iou threshold for each chunk.
        :param data (dict): containing data from the training loop
        :return: updated data
        """
        clips_in_chunk = self.window_size // self.num_chunks # number of clips in each chunk
        vis_feats = data[self.in_name] # B T C
        B, T, C = vis_feats.shape
        indices = torch.arange(T).unsqueeze(0).repeat(B,1).float() # B T
        indices = indices.unsqueeze(2).unsqueeze(1) # B 1 T 1
        # unfold visual feats
        ks = (self.window_size, 1)
        st, pd, dl = (1, 1), (0, 0), (1, 1)
        indices_unfolded = F.unfold(indices, ks, dl, pd, st) # B KS NW
        _, _, NW = indices_unfolded.shape
        indices_unfolded = indices_unfolded.transpose(1,2).view(B, NW, self.num_chunks, clips_in_chunk)
        print("indices unfolded shape:",indices_unfolded.shape)
        data[self.out_name] = indices_unfolded
        return data
