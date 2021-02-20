import numpy as np
import torch
import json

class TVRH5StartEnd:
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
    def __init__(self, start_name, end_name, duration, output):
        """
        """
        self.start_name = start_name
        self.end_name = end_name
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
        NS = data[self.start_name].shape[1]
        NE = data[self.end_name].shape[1]
        starts = torch.argsort(data[self.start_name],dim=1,descending=True)
        ends = torch.argsort(data[self.end_name],dim=1,descending=True)
        for b in range(starts.shape[0]):
            vid_name = data["vid_name"][b]
            if vid_name not in self.video2idx.keys():
                self.video2idx[vid_name] = self.vid_index
                self.vid_index = self.vid_index + 1
            sts = starts[b,:,0].data.cpu().numpy() / NS * data[self.duration][b]
            ens = (1+ends[b,:,0].data.cpu().numpy()) / NE * data[self.duration][b] # inclusive ends
            all_preds = [ [self.video2idx[vid_name],
                           min(sts[ii],ens[ii]),
                           max(sts[ii],ens[ii]),
                            1.0] for ii in range(NS)]
            next_pred = {
                         "desc_id":data["desc_id"][b],
                         "desc":data["desc"][b],
                         "predictions":all_preds
                        }
            self.pred.append(next_pred)
            # print(next_pred["predictions"])

    def conclude(self):
        """
        save the predictions to json
        """
        results = {"video2idx":self.video2idx,"SVMR":self.pred}
        with open(self.output,"wt") as f:
            f.write(json.dumps(results))

    def __str__(self):
        return ""