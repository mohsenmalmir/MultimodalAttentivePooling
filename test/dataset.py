import unittest
import torch
from multimodalattentivepooling.dataset.tvrh5 import TVRH5
from pathlib import Path

class TestDatasets(unittest.TestCase):

    def test_tvrh5(self):
        # create visual words, textual words
        json_file = Path("/Users/mohsenmalmir/Code/Vision/TVRetrieval/data/tvr_train_release.jsonl")
        vis_h5file = Path("/Users/mohsenmalmir/Documents/data/tvr_feature_release/video_feature/tvr_i3d_rgb600_avg_cl-1.5.h5")
        query_h5file = Path("/Users/mohsenmalmir/Documents/data/tvr_feature_release/bert_feature/query_only/tvr_query_pretrained_w_query.h5")
        ds = TVRH5(json_file, vis_h5file, query_h5file)
        print("num records in dataset:",len(ds))
        for ii, d in enumerate(ds):
            if ii>1:
                break
            print(d.keys(),d["query_feats"].shape, d["duration"], d["vis_feats"].shape)


if __name__ == '__main__':
    unittest.main()