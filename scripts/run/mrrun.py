import argparse
from pathlib import Path
from multimodalattentivepooling.dataset.momentretrieval import  MomentRetrieval
from multimodalattentivepooling.model.attentivepooling import CorpusAttentivePool

def create_parse():
    parser = argparse.ArgumentParser(description='Running moment retrieval training loop.')
    parser.add_argument('--train-json', type=Path, required=True, help='json file containing training moments')
    parser.add_argument('--val-json', type=Path, required=True, help='json file containing validation moments')
    parser.add_argument('--img-dir', type=Path, required=True, help='path containing images')
    parser.add_argument('--ds-transform', nargs="+", type=str, required=True, help='preprocessing transformations applied to data')

    return parser



def run(config):
    pass

if __name__=="__main__":
    parser = create_parse()
    args = parser.parse_args()
    run(config=args.config)
