import argparse
from pathlib import Path
import numpy as np
from multimodalattentivepooling.utils.modload import load_comp, load_args

def create_parse():
    parser = argparse.ArgumentParser(description='Prepare TVR dataset.')
    parser.add_argument('--dataset', '-ds', type=str, required=True, help='dataset module')
    parser.add_argument('--dataset-args', '-dsa', type=Path, required=True, help='dataset args YAML')
    parser.add_argument('--dataloader', '-dl', type=str, required=True, help='dataloader module')
    parser.add_argument('--dataloader-args', '-dla', type=Path, required=True, help='dataloader args YAML')
    parser.add_argument('--optimizer', '-o', type=str, required=True, help='optimizer')
    parser.add_argument('--optimizer-args', '-oa', type=Path, required=False, help='optimizer args YAML file')
    parser.add_argument('--net', '-n', type=str, required=True, help='network module')
    parser.add_argument('--net-args', '-na', type=Path, required=True, help='network args YAML')
    parser.add_argument('--loss', '-l', type=str, required=True, help='loss module')
    parser.add_argument('--loss-args', '-la', type=Path, required=True, help='loss args YAML')
    return parser


def run(dataset, dataset_args, dataloader, dataloader_args, optimizer, optimizer_args, net, net_args, loss, loss_args):
    """

    :param dataset (str): module specification of the dataset
    :param dataset_args: YAML file containing dataset arguments
    :param dataloader: module specification of the dataloader, for example torch.utils.data.DataLoader
    :param dataloader_args: YAML file containing dataloader arguments. The dataset will also be passed to this class.
    :param optimizer: module for optimizer, e.g. torch.optim.Adam. network params will be pass to this class
    :param optimizer_args: YAML file containing optimizer params
    :param net: module specification for network
    :param net_args:
    :param loss:
    :param loss_args:
    :return:
    """
    # load components dynamically, initialize them
    dataset, dataset_args = load_comp(dataset), load_args(dataset_args)
    dataset = dataset(**dataset_args)
    dataloader, dataloader_args = load_comp(dataloader), load_args(dataloader_args)
    dataloader = dataloader(dataset, **dataloader_args)
    print("data loader:",len(dataloader))
    net, net_args = load_comp(net), load_args(net_args)
    net = net(**net_args)
    print("created network with {0} parameters".format(sum([np.prod(p.shape) for p in net.parameters() if p.requires_grad])))
    optimizer, optimizer_args = load_comp(optimizer), load_args(optimizer_args)
    optimizer = optimizer(net.parameters(),**optimizer_args)


if __name__=="__main__":
    parser = create_parse()
    args = parser.parse_args()
    run(**vars(args))