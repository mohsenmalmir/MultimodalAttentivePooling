import torch
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
        # print(data[self.out_name].shape, data[self.tgt_name].shape)
        data["loss"] = self.loss(data[self.out_name], data[self.tgt_name])
        return data

