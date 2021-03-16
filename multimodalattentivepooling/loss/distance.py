import torch
import numpy as np
from torch.nn import BCEWithLogitsLoss
from multimodalattentivepooling.utils.moduleload import load_comp, load_args


class Dist:
    """
    implements Dist (L1,L2,etc) with ignore_value
    """
    def __init__(self, **kwargs):
        """
        :return:
        """
        mod_info = kwargs["module"]
        loss = load_comp(mod_info["comp"])
        for k, v in mod_info["args"].items():
            mod_info["args"][k] = torch.tensor(v)
        self.loss = loss(**mod_info["args"])
        self.out_name = kwargs["out_name"]
        self.tgt_name = kwargs["tgt_name"]
        self.ignore_val = kwargs["ignore_val"]

    def __call__(self, data):
        """
        This function receives an input dictionary containing output and target value. It calculates the loss
        and returns the value.
        :param data(dict): containing network output and target
        :return: updated data, with "loss"
        """
        if "loss" not in data.keys():
            data["loss"] = 0
        mask = torch.where(data[self.tgt_name]!=self.ignore_val,1,0)
        data["loss"] = data["loss"] + self.loss(mask*data[self.out_name], mask*data[self.tgt_name])
        return data

    def to(self, device):
        self.loss.to(device)
