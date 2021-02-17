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
        videos = json.load(f)
    print("extracting moments")
    moments = []
    with h5py.File(output, "w") as F:
        for vid_name in videos.keys():
            desc_cntr = 0
            grp = F.create_group(vid_name)
            for ii in range(len(videos[vid_name]["sentences"])):
                mmt = dict()
                mmt["vid_name"] = vid_name
                mmt["desc_id"] = desc_cntr
                mmt["ts"] = videos[vid_name]["timestamps"][ii]
                mmt["desc"] = videos[vid_name]["sentences"][ii]
                mmt["duration"] = videos[vid_name]["duration"]
                moments.append(mmt)
                desc_cntr += 1
                # bert encoding
                input_ids = torch.tensor(tokenizer.encode(mmt["desc"])).unsqueeze(0).to(device)  # Batch size 1
                outputs = model(input_ids)
                last_hidden_states = outputs[0][0].data.cpu().numpy()
                grp.create_dataset(str(desc_cntr),data=last_hidden_states)
    print("{0} moments loaded".format(len(moments)))



if __name__=="__main__":
    parser = create_parse()
    args = parser.parse_args()
    run(**vars(args))