import argparse
from pathlib import Path
import importlib
import yaml

def create_parse():
    parser = argparse.ArgumentParser(description='Prepare TVR dataset.')
    parser.add_argument('--dataset', '-ds', type=str, required=True, help='dataset module')
    parser.add_argument('--dataset-args', '-dsa', type=Path, required=True, help='dataset args YAML')
    parser.add_argument('--dataloader', '-dl', type=str, required=True, help='dataloader module')
    parser.add_argument('--dataloader-args', '-dla', type=Path, required=True, help='dataloader args YAML')
    parser.add_argument('--optimizer', '-o', type=str, required=True, help='optimizer')
    parser.add_argument('--optimizer-args', '-oa', type=Path, required=True, help='optimizer args YAML file')
    parser.add_argument('--net', '-n', type=str, required=True, help='network module')
    parser.add_argument('--net-args', '-na', type=Path, required=True, help='network args YAML')
    parser.add_argument('--loss', '-l', type=str, required=True, help='loss module')
    parser.add_argument('--loss-args', '-la', type=Path, required=True, help='loss args YAML')
    return parser

def load_comp(comp_full_path):
    """
    given the full path to a class, this function will load the class dynamically.
    :param comp_full_path: full path of the module, example: torch.nn.Linear
    :return: class object
    """
    path_split = comp_full_path.split(".")
    pkg, comp = ".".join(path_split[:-1]), path_split[-1]
    pkg = importlib.import_module(pkg)
    comp = getattr(pkg, comp)
    return comp

def load_args(args_file):
    """
    Given a Path object to a YAML file, this function will load the file into a dictionary and return the result.
    :param args_file: Path to the YAML file
    :return: dictionary
    """
    with open(args_file,"rt") as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    return args

def run(dataset, dataset_args, dataloader, dataloader_args, optimizer, optimizer_args, net, net_args, loss, loss_args):
    # load components dynamically
    dataset, dataset_args = load_comp(dataset), load_args(dataset_args)
    dataset = dataset(**dataset_args)
    print(len(dataset))


if __name__=="__main__":
    parser = create_parse()
    args = parser.parse_args()
    run(**vars(args))