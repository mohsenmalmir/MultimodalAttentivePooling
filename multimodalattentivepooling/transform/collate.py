import numpy as np
import torch

class CollateTVRH5:
    """
    Given a set of feature vectors, this class collates the feature vectors into a single tensor.
    """
    def __init__(self):
        pass

    def __call__(self, data):
        # print([type(d["vis_feats"]) for d in data])
        vis_len = [d["vis_feats"].shape[0] for d in data] # T C
        VD = data[0]["vis_feats"].shape[1] # VID feat size
        query_len = [d["query_feats"].shape[0] for d in data] # T C
        QD = data[0]["vis_feats"].shape[1] # Query feat size
        # find max vid len, padd
        mx_vid_len = max(max(vis_len),66)
        vis_feats = [np.pad(d["vis_feats"],((0,mx_vid_len-l),(0,0)))[np.newaxis,...] for d,l in zip(data,vis_len)]
        vis_feats = np.concatenate(vis_feats,axis=0)
        # print(vis_feats.shape)
        # find max seq len, padd
        mx_query_len = max(query_len)
        query_feats = [np.pad(d["query_feats"],((0,mx_query_len-l),(0,0)))[np.newaxis,...] for d,l in zip(data,query_len)]
        query_feats = np.concatenate(query_feats,axis=0)
        return {"vis_feats":torch.tensor(vis_feats).float(),
                "query_feats":torch.tensor(query_feats).float(),
                "ts":[d["ts"] for d in data],
                "duration":[d["duration"] for d in data],
                "vis_len":vis_len,
                "query_len":query_len,
                "desc":[d["desc"] for d in data],
                "vid_name": [d["vid_name"] for d in data],
                "desc_id": [d["desc_id"] for d in data],
                }