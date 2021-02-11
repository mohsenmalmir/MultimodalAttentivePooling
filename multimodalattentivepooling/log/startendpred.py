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
        starts = torch.argmax(data[self.start_name],dim=1)
        ends = torch.argmax(data[self.end_name],dim=1)
        for b in range(starts.shape[0]):
            vid_name = data["vid_name"][b]
            if vid_name not in self.video2idx.keys():
                self.video2idx[vid_name] = self.vid_index
                self.vid_index = self.vid_index + 1
            st = starts[b,0].item() / NS * data[self.duration][b]
            en = ends[b,0].item() / NE * data[self.duration][b]
            next_pred = {
                         "desc_id":data["desc_id"][b],
                         "desc":data["desc"][b],
                         "predictions":[self.video2idx[vid_name],min(st,en),max(st,en),1.0]
                        }
            # print(next_pred)
            self.pred.append(next_pred)

    def wrap_up(self):
        """
        save the predictions to json
        """
        results = {"video2idx":self.video2idx,"SVMR":self.pred}
        with open(self.output,"wt") as f:
            f.write(json.dumps(results))

    def __str__(self):
        return ""