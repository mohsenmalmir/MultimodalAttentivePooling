import json
import torch
import h5py
from torch.utils.data import Dataset
import numpy as np


class TVRH5(Dataset):
    """
    This class implements a generic moment retrieval dataset.
    The dataset is organized around a set of frames and a json file that describes moments.
    """
    def __init__(self, json_file, feats_h5file, query_h5file):
        """
        Args:
            :param json_file(str): that includes list of videos, queries/time span per video.
            :param feats_h5file(str): path to the features file
            :param query_h5file(str): path to the query file
        """
        with open(json_file,"rt") as f:
            self.moments = [json.loads(l) for l in f]
        self.vis_feats = h5py.File(feats_h5file,"r")
        self.query_feats = h5py.File(query_h5file,"r")
        print("preloading...")
        # self.vis_feats_np = dict()
        # self.query_feats_np = dict()
        # for ii,k in enumerate(self.moments):
        #     if ii%1000==0:
        #         print(ii)
        #     self.vis_feats_np[k["vid_name"]] = np.array(self.vis_feats[k["vid_name"]]).shape
        #     self.query_feats_np[str(k["desc_id"])] = np.array(self.query_feats[str(k["desc_id"])])

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
                # "query_text":m["desc"],
                "duration":m["duration"]
                }
        return data