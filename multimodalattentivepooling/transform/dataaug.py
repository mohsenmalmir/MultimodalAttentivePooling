import numpy as np
import torch

class DataAug:
    """
    transform an input sequence by cropping from beginning or end.
    """
    def __init__(self, max_moment_clip=0.1, max_clip=0.3):
        self.max_moment_clip = max_moment_clip
        self.max_clip = max_clip

    def __call__(self, data):
        # print(data["duration"],data["ts"],data["vis_len"],data["vis_feats"].shape)
        # data is a single data point
        # augment half of the times
        if np.random.uniform(0,1)<0.5:
            if np.random.uniform(0,1)<0.5:# crop head
                # this is cropping point where at most 10% of the moment will be clipped
                start_limit = data["ts"][0] + self.max_moment_clip * (data["ts"][1]-data["ts"][0])
                new_start = min(start_limit,(np.random.uniform(0,self.max_clip) * data["duration"]))
                frame_start = int(new_start / data["duration"] * data["vis_len"])
                if frame_start > 0:
                    data["duration"] = data["duration"] - new_start
                    data["vis_feats"] = data["vis_feats"][frame_start:,:]
                    data["ts"] = [max(data["ts"][0]-new_start,0),data["ts"][1]-new_start]
                    data["vis_len"] = data["vis_feats"].shape[0]
                    # print("START")
            else: # crop tail
                end_limit = data["ts"][1] - self.max_moment_clip * (data["ts"][1]-data["ts"][0])
                new_end = max(end_limit,(np.random.uniform(1-self.max_clip,1) * data["duration"]))
                frame_end = int(new_end / data["duration"] * data["vis_len"])
                if frame_end<data["vis_len"]:
                    data["vis_feats"] = data["vis_feats"][:frame_end,:]
                    data["duration"] = new_end
                    data["ts"][1] = min(data["ts"][1],new_end)
                    data["vis_len"] = data["vis_feats"].shape[0]
                    # print("END")
            # print("----->",data["duration"],data["ts"],data["vis_len"],data["vis_feats"].shape)
        return data
