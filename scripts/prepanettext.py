import argparse
import json
import numpy as np
import torch
import h5py
from transformers import BertTokenizer, BertModel

def create_parse():
    parser = argparse.ArgumentParser(description='Prepare TVR dataset.')
    parser.add_argument('--input', '-i', type=str, required=True, help='path to json file containing anet data')
    parser.add_argument('--output', '-o', type=str, required=True, help='output hdf5')
    parser.add_argument('--device', '-d', type=str, required=True, help='device')
    return parser


def run(input, output, device):
    """
    :param input: path to json file
    :param output: path to hdf5 file
    :return:
    """
    device = torch.device(device)
    print("loading bert...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased').to(device)
    print("loading json input")
    with open(input,"rt") as f:
        moments = [json.loads(l) for l in f]
    print("{0} moments loaded".format(len(moments)))
    with h5py.File(output, "w") as F:
        for ii,m in enumerate(moments):
            if ii%1000==0:
                print(ii)
            # bert encoding
            input_ids = torch.tensor(tokenizer.encode(m["desc"])).unsqueeze(0).to(device)  # Batch size 1
            outputs = model(input_ids)
            last_hidden_states = outputs[0][0].data.cpu().numpy()
            F.create_dataset(str(m["desc_id"]),data=last_hidden_states)



if __name__=="__main__":
    parser = create_parse()
    args = parser.parse_args()
    run(**vars(args))