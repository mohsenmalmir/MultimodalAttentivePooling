import numpy as np
import torch
import json
import torch.nn.functional as F

class TVRH5Segment:
    """
    save the output of model for tvrh5 data
{
    "video2idx": {
        "castle_s01e02_seg02_clip_09": 19614,
        ...
    },
    "SVMR": [{
            "desc_id": 90200,
            "desc": "Phoebe puts one of her ponytails in her mouth.",
            "predictions": [
                [20092, 36.0, 42.0, -1.9082],
                [20092, 18.0, 24.0, -1.9145],
                [20092, 51.0, 54.0, -1.922],
                ...
            ]
        },
        ...
    ],
    }
    """
    def __init__(self, seg_name, duration, output):
        """
        """
        self.seg_name = seg_name
        self.duration = duration
        self.output = output
        self.vid_index = 0
        self.video2idx = dict()
        self.pred = []


    def __call__(self, data):
        """
        for the given data, save the output results of network to json file.
        :param data:
        :return:
        """
        segmented = data[self.seg_name]
        NE = segmented.shape[2]
        NS = NE
        segmented = F.softmax(segmented,dim=1)[:,1,:].data.cpu().numpy()
        print(segmented>0.5)
        segmented = [np.where(segmented[b,:]>0.5)[0] for b in range(segmented.shape[0])]
        starts = np.array([S[0] if len(S)>0 else 0 for S in segmented])
        ends = np.array([S[-1] if len(S)>0 else NS for S in segmented])
        for b in range(starts.shape[0]):
            vid_name = data["vid_name"][b]
            if vid_name not in self.video2idx.keys():
                self.video2idx[vid_name] = self.vid_index
                self.vid_index = self.vid_index + 1
            sts = starts[b] / NS * data[self.duration][b]
            ens = (1+ends[b]) / NE * data[self.duration][b] # inclusive ends
            all_preds = [ [self.video2idx[vid_name],
                           min(sts,ens),
                           max(sts,ens),
                            1.0] ]
            next_pred = {
                         "desc_id":data["desc_id"][b],
                         "desc":data["desc"][b],
                         "predictions":all_preds
                        }
            self.pred.append(next_pred)
            print(next_pred["predictions"])

    def wrap_up(self):
        """
        save the predictions to json
        """
        results = {"video2idx":self.video2idx,"SVMR":self.pred}
        with open(self.output,"wt") as f:
            f.write(json.dumps(results))

    def __str__(self):
        return ""


class LogSegment:
    """
    save the output of model for segment prediction
{
    "video2idx": {
        "castle_s01e02_seg02_clip_09": 19614,
        ...
    },
    "SVMR": [{
            "desc_id": 90200,
            "desc": "Phoebe puts one of her ponytails in her mouth.",
            "predictions": [
                [20092, 36.0, 42.0, -1.9082],
                [20092, 18.0, 24.0, -1.9145],
                [20092, 51.0, 54.0, -1.922],
                ...
            ]
        },
        ...
    ],
    }
    """
    def __init__(self, pred_name, dur_name, len_name, vid_name, out_name):
        """
        """
        self.pred_name = pred_name
        self.dur_name = dur_name
        self.len_name = len_name
        self.vid_name = vid_name
        self.out_name = out_name
        self.vid_index = 0
        self.video2idx = dict()
        self.pred = []


    def __call__(self, data):
        """
        for the given data, save the output results of network to json file.
        :param data:
        :return:
        """
        segmented = data[self.pred_name]
        # print([S for S in segmented])
        starts = np.array([S[0] if S[0]!=-1 else 0 for S in segmented])
        ends = np.array([S[1] if S[1]!=-1 else 0 for S in segmented])
        for b in range(starts.shape[0]):
            vid_name = data[self.vid_name][b]
            if vid_name not in self.video2idx.keys():
                self.video2idx[vid_name] = self.vid_index
                self.vid_index = self.vid_index + 1
            sts = starts[b] / data[self.len_name][b] * data[self.dur_name][b]
            ens = (1+ends[b]) / data[self.len_name][b] * data[self.dur_name][b] # inclusive ends
            all_preds = [ [self.video2idx[vid_name],
                           min(sts,ens),
                           max(sts,ens),
                           1.0] ]
            next_pred = {
                "desc_id":data["desc_id"][b],
                "desc":data["desc"][b],
                "predictions":all_preds
            }
            self.pred.append(next_pred)
            # print(next_pred["predictions"])

    def wrap_up(self):
        """
        save the predictions to json
        """
        results = {"video2idx":self.video2idx,"SVMR":self.pred}
        with open(self.output,"wt") as f:
            f.write(json.dumps(results))

    def __str__(self):
        return ""