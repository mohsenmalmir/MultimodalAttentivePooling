import importlib
import yaml

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
    if args_file==None:
        return dict()
    with open(args_file,"rt") as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    return args
