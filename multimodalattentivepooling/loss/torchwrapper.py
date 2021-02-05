import torch
import numpy as np
from torch.nn import BCEWithLogitsLoss
from multimodalattentivepooling.utils.moduleload import load_comp, load_args


class TorchLossWrapper:
    """
    This is a wrapper around torch loss classes. It initializes the torch loss with the given parameters.
    """
    def __init__(self, **kwargs):
        """
        initialize the loss wrapper.
        :param kwargs: {"module":{"comp":name,"arg1":val1,...},}
        :return:
        """
        mod_info = kwargs["module"]
        loss = load_comp(mod_info["comp"])
        for k, v in mod_info["args"].items():
            mod_info["args"][k] = torch.tensor(v)
        self.loss = loss(**mod_info["args"])
        self.out_name = kwargs["out_name"]
        self.tgt_name = kwargs["tgt_name"]

    def __call__(self, data):
        """
        This function receives an input dictionary containing output and target value. It calculates the loss
        and returns the value.
        :param data(dict): containing network output and target
        :return: updated data, with "loss"
        """
        if "loss" not in data.keys():
            data["loss"] = 0
        data["loss"] = data["loss"] + self.loss(data[self.out_name], data[self.tgt_name])
        return data

    def to(self, device):
        self.loss.to(device)

class TorchLossArray:
    """
    implements an array of torch losses with specific parameters.
    """
    def __init__(self, **kwargs):
        self.loss = dict()
        self.pred_name, self.tgt_name = dict(), dict()
        for mod_name in kwargs["TorchArrayLoss"].keys():
            mod = kwargs["TorchArrayLoss"][mod_name]
            loss = load_comp(mod["comp"])
            for k, v in mod["args"].items():
                mod["args"][k] = torch.tensor(v)
            self.loss[mod_name] = loss(**mod["args"])
            self.pred_name[mod_name] = mod["pred_name"]
            self.tgt_name[mod_name] = mod["tgt_name"]

    def __call__(self, data):
        """
        This function receives an input dictionary containing output and target value. It calculates the loss
        and returns the value.
        :param data(dict): containing network output and target
        :return: updated data, with "loss"
        """
        if "loss" not in data.keys():
            data["loss"] = 0
        L = data["loss"]
        for mod_name in self.loss.keys():
            L = L + self.loss[mod_name](data[self.pred_name[mod_name]], data[self.tgt_name[mod_name]])
            # print(mod_name,L)
        data["loss"] = L
        return data

    def to(self, device):
        for mod_name in self.loss.keys():
            self.loss[mod_name].to(device)

class TorchLossSamplingArray:
    """
    implements an array of torch losses with specific parameters.
    """
    def __init__(self, **kwargs):
        self.loss = dict()
        self.pred_name, self.tgt_name = dict(), dict()
        self.sample_ratio = dict()
        for mod_name in kwargs["TorchArrayLoss"].keys():
            mod = kwargs["TorchArrayLoss"][mod_name]
            loss = load_comp(mod["comp"])
            for k, v in mod["args"].items():
                mod["args"][k] = torch.tensor(v)
            self.loss[mod_name] = loss(**mod["args"])
            self.pred_name[mod_name] = mod["pred_name"]
            self.tgt_name[mod_name] = mod["tgt_name"]
            self.sample_ratio[mod_name] = mod["sample_ratio"]

    def __call__(self, data):
        """
        This function receives an input dictionary containing output and target value. It calculates the loss
        and returns the value.
        :param data(dict): containing network output and target
        :return: updated data, with "loss"
        """
        if "loss" not in data.keys():
            data["loss"] = 0
        L = data["loss"]
        for mod_name in self.loss.keys():
            # sample target
            tgt = torch.nonzero(data[self.tgt_name[mod_name]]==1,as_tuple=True)
            if tgt[0].shape[0]>0:
                n_samples = int(self.sample_ratio[mod_name]*tgt[0].shape[0]) # number of 1s
                idx = np.arange(tgt[0].shape[0])
                np.random.shuffle(idx)
                idx = idx[:n_samples+1]
                tgt = tuple(t[idx] for t in tgt)
                LTGT = self.loss[mod_name](data[self.pred_name[mod_name]][tgt], data[self.tgt_name[mod_name]][tgt])
                L = L + LTGT
            # sample background
                bg = torch.nonzero(data[self.tgt_name[mod_name]]==0,as_tuple=True)
                if bg[0].shape[0]>0:
                    idx = np.arange(bg[0].shape[0])
                    np.random.shuffle(idx)
                    idx = idx[:n_samples+1]
                    bg = tuple(b[idx] for b in bg)
                    LBG = self.loss[mod_name](data[self.pred_name[mod_name]][bg], data[self.tgt_name[mod_name]][bg])
                    L = L + LBG
        data["loss"] = L
        return data

    def to(self, device):
        for mod_name in self.loss.keys():
            self.loss[mod_name].to(device)
