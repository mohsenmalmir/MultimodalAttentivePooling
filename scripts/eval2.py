import argparse
import numpy as np
import torch
from multimodalattentivepooling.utils.moduleload import load_comp, load_args

def create_parse():
    parser = argparse.ArgumentParser(description='Prepare TVR dataset.')
    parser.add_argument('--config', '-cfg', type=str, required=True, help='dataset module')
    return parser


def run(dataset, dataset_args, dataloader, dataloader_args, transforms, transforms_args, net, net_args, eval_args,
        logger, logger_args):
    """
    :param dataset (str): module specification of the dataset
    :param dataset_args: YAML file containing dataset arguments
    :param dataloader: module specification of the dataloader, for example torch.utils.data.DataLoader
    :param dataloader_args: YAML file containing dataloader arguments. The dataset will also be passed to this class.
    :param transforms: composite transforms class
    :param transforms_args: YAML file containing transforms arguments (pipeline of transform functions)
    :param net: module specification for network
    :param net_args: YAML file specifying network arguments
    :param eval_args: YAML file containing training args
    :param logger: logger module
    :param logger_args: YAML file containing logger args
    :return:
    """
    # load components dynamically, initialize them
    dataset, dataset_args = load_comp(dataset), load_args(dataset_args)
    dataset = dataset(**dataset_args)
    dataloader, dataloader_args = load_comp(dataloader), load_args(dataloader_args)
    dataloader = dataloader(dataset, **dataloader_args)
    print("data loader:",len(dataloader))
    transforms, transforms_args= load_comp(transforms), load_args(transforms_args)
    transforms = transforms(**transforms_args)
    net, net_args = load_comp(net), load_args(net_args)
    net = net(**net_args)
    print("created network with {0} parameters".format(sum([np.prod(p.shape) for p in net.parameters() if p.requires_grad])))
    eval_args = load_args(eval_args)
    logger, logger_args = load_comp(logger), load_args(logger_args)
    logger = logger(**logger_args)
    # training loop
    device = torch.device(eval_args["device"]["name"])
    net.to(device)
    print("loading from checkpoint:",device)
    net.load_state_dict(torch.load(eval_args["ckpt"], map_location=device))
    net.eval()
    for epoch_index, data in enumerate(dataloader):
        # pass data through transforms
        data = transforms(data)
        for n in eval_args["device"]["data"]:
            data[n] = data[n].to(device)
        # network output
        data = net(data)
        # log
        logger(data)
        print(epoch_index)
        # print(data["sttgt_maxpooled"].tolist())
        # print(torch.argmax(data["start_maxpooled"],dim=1).tolist())
        # print(data["endtgt_maxpooled"].tolist())
        # print(torch.argmax(data["end_maxpooled"],dim=1).tolist())
        # print(data["ts"])
        # print(data["win3gt"])

    logger.conclude()


if __name__=="__main__":
    parser = create_parse()
    args = parser.parse_args()
    # load run config
    args = load_args(args.config)
    run(**args)