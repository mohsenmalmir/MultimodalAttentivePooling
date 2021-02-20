import argparse
import json

def create_parse():
    parser = argparse.ArgumentParser(description='Prepare TVR dataset.')
    parser.add_argument('--input', '-i', type=str, required=True, help='input activitynet json file')
    parser.add_argument('--output', '-o', type=str, required=True, help='output TVR json file')
    return parser


def run(input, output):
    """
    convert an activitynet json to TVR json.
    :param input:
    :param output:
    :return:
    """
    with open(input,"r") as f:
        anet_data = json.load(f)
    desc_id = 0
    tvr_format = []
    for vid_name in anet_data.keys():
        for ii in range(len(anet_data[vid_name]["sentences"])):
            mmt = dict()
            mmt["vid_name"] = vid_name
            mmt["desc_id"] = desc_id
            mmt["ts"] = anet_data[vid_name]["timestamps"][ii]
            mmt["desc"] = anet_data[vid_name]["sentences"][ii]
            mmt["duration"] = anet_data[vid_name]["duration"]
            mmt["type"] = "v" # this is not used but required for TVR eval script
            desc_id += 1
            tvr_format.append(mmt)
    with open(output,"wt") as f:
        for m in tvr_format:
            f.write(json.dumps(m)+"\n")




if __name__=="__main__":
    parser = create_parse()
    args = parser.parse_args()
    run(**vars(args))