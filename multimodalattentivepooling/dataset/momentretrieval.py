import json
import torch
from pathlib import Path
from torch.utils.data import Dataset
from skimage import io, transform


class MomentRetrieval(Dataset):
    """
    This class implements a generic moment retrieval dataset.
    The dataset is organized around a set of frames and a json file that describes moments.
    """
    def __init__(self, json_file, img_dir, transform=None):
        """
        Args:
            :param json_file(Path): that includes list of videos, queries/time span per video.
            :param img_dir(Path): root dir containing frames
            :param transform(list): list of transform functions for the dataset
        """
        with open(json_file.as_posix(),"rt") as f:
            self.moments = [json.loads(l) for l in f]
        self.img_dir = img_dir
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
        imgs_path = self.img_dir / m["rel_path"]
        image_names = imgs_path.glob("*.jpg")
        images = list(map(io.imread,image_names))
        data = {"image": images,
                "ts_frame": m["ts_frame"],
                "subs":[s["text"] for s in m["subtitles"]],
                "query":m["desc"],
                "frame_label":m["frame_label"],
                "frame_label_pos_weights":m["frame_pos_weights"],
                }
        # this is to enable a cascade of transforms that are modular, e.g. resize, augment, word2vec, etc.
        for t in self.transform:
            data = t(data)
        return data

