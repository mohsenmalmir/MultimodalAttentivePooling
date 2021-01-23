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
    def __init__(self, json_file, feats_h5file, query_h5file, transform=None):
        """
        Args:
            :param json_file(Path): that includes list of videos, queries/time span per video.
            :param feats_h5file(Path): path to the features file
            :param query_h5file(Path): path to the query file
            :param transform(list): list of transform functions for the dataset
        """
        with open(json_file.as_posix(),"rt") as f:
            self.moments = [json.loads(l) for l in f]
        self.vis_feats = h5py.File(str(feats_h5file),"r")
        self.query_feats = h5py.File(str(query_h5file),"r")
        self.transform = transform

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
                "query_text":m["desc"],
                "duration":m["duration"]
                }
        # this is to enable a cascade of transforms that are modular, e.g. resize, augment, word2vec, etc.
        if self.transform:
            for t in self.transform:
                data = t(data)
        return data