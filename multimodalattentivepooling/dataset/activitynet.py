import json
import torch
import h5py
from torch.utils.data import Dataset
import numpy as np


class ActivityNet(Dataset):
    """
    activitynet c3d features from h5py file.
    """
    def __init__(self, json_file, feats_h5file, query_h5file):
        """
        Args:
            :param json_file(str): that includes list of videos, queries/time span per video.
            :param feats_h5file(str): path to the features file
            :param query_h5file(str): path to the query file
        """
        with open(json_file,"rt") as f:
            videos = json.load(f)
        self.moments = []
        for vid_name in videos.keys():
            for ii in range(len(videos[vid_name]["sentences"])):
                mmt = dict()
                mmt["vid_name"] = vid_name
                mmt["desc_id"] = len(self.moments)
                mmt["ts"] = videos[vid_name]["timestamps"][ii]
                mmt["desc"] = videos[vid_name]["sentences"][ii]
                mmt["duration"] = videos[vid_name]["duration"]
                self.moments.append(mmt)
        self.vis_feats = h5py.File(feats_h5file,"r")
        self.query_feats = h5py.File(query_h5file,"r")

    def __len__(self):
        """
        return dataset size
        """
        return len(self.moments)

    def __getitem__(self, idx):
        """
        get the next item in the dataset, which is a video/query pair
        :param idx: index of the data to retrieve
        :return:
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # retrieve moment
        m = self.moments[idx]
        # load all images of this clip
        data = {
                "vis_feats": np.array(self.vis_feats[m["vid_name"]]),
                "query_feats": np.array(self.query_feats[str(m["desc_id"])]),
                "ts": m["ts"],
                "desc":m["desc"],
                "duration":m["duration"],
                "vid_name": m["vid_name"],
                "desc_id": m["desc_id"],
                }
        return data