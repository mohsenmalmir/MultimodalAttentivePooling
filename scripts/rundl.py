import argparse
from pathlib import Path
import numpy as np
from multimodalattentivepooling.utils.moduleload import load_comp, load_args

def create_parse():
    parser = argparse.ArgumentParser(description='Prepare TVR dataset.')
    parser.add_argument('--dataset', '-ds', type=str, required=True, help='dataset module')
    parser.add_argument('--dataset-args', '-dsa', type=Path, required=True, help='dataset args YAML')
    parser.add_argument('--dataloader', '-dl', type=str, required=True, help='dataloader module')
    parser.add_argument('--dataloader-args', '-dla', type=Path, required=True, help='dataloader args YAML')
    parser.add_argument('--transforms', '-t', type=str, required=True, help='data transforms module')
    parser.add_argument('--transforms-args', '-tfa', type=Path, required=True, help='YAML file containing transforms args')
    return parser


def run(dataset, dataset_args, dataloader, dataloader_args, transforms, transforms_args):
    """
    :param dataset (str): module specification of the dataset
    :param dataset_args: YAML file containing dataset arguments
    :param dataloader: module specification of the dataloader, for example torch.utils.data.DataLoader
    :param dataloader_args: YAML file containing dataloader arguments. The dataset will also be passed to this class.
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
    # extract data
    bc = np.zeros(2)
    for epoch_index, data in enumerate(dataloader):
        if epoch_index%1000==0:
            print(epoch_index)
        data = transforms(data)
        gt = data["gt"].data.cpu().numpy().astype(int)
        bc = bc + np.bincount(gt[np.where(gt > -1)])
    print(bc)



if __name__=="__main__":
    parser = create_parse()
    args = parser.parse_args()
    run(**vars(args))