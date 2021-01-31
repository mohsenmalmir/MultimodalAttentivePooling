import argparse
from pathlib import Path
import numpy as np
from multimodalattentivepooling.utils.moduleload import load_comp, load_args

def create_parse():
    parser = argparse.ArgumentParser(description='Prepare TVR dataset.')
    parser.add_argument('--dataset', '-ds', type=str, required=True, help='dataset module')
    parser.add_argument('--dataset-args', '-dsa', type=Path, required=True, help='dataset args YAML')
    return parser


def run(dataset, dataset_args):
    """
    :param dataset (str): module specification of the dataset
    :param dataset_args: YAML file containing dataset arguments
    :return:
    """
    # load components dynamically, initialize them
    dataset, dataset_args = load_comp(dataset), load_args(dataset_args)
    dataset = dataset(**dataset_args)
    print("num data points:",len(dataset))
    winsizes = []
    for ii, d in enumerate(dataset):
        if ii%1000==0:
            print(ii)
        nframes = d["vis_feats"].shape[0]
        ts = d["ts"]
        dur = d["duration"]
        winsize = (ts[1]-ts[0])/dur * nframes
        winsizes.append(winsize)
        if ii>5000:
            break
    p = np.linspace(0, 100, 100)
    print(np.percentile(winsizes, p))
    # print(np.histogram(winsizes,bins=10))


if __name__=="__main__":
    parser = create_parse()
    args = parser.parse_args()
    run(**vars(args))