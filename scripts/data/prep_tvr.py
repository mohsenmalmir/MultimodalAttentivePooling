import argparse
import json
from pathlib import Path
import numpy as np
from multimodalattentivepooling.eval.utils import intersect

def create_parse():
    parser = argparse.ArgumentParser(description='Prepare TVR dataset.')
    parser.add_argument('--input-json', '-i', type=Path, required=True, help='input json file from tvr git')
    parser.add_argument('--img-dir', '-d', type=Path, required=True, help='root containing tvr frames')
    parser.add_argument('--subtitles', '-s', type=Path, required=True, help='path to the subtitles json')
    parser.add_argument('--output', '-o', type=Path, required=True, help='output json file')
    return parser

def run(input_json, img_dir, subtitles, output):
    """
    given an input json file from tvr dataset, this function will check the existence of videos
    for all moments in the input json file. It will then create a new json file where videos, queries and
    moments mapped to frames are stored.
    :param input_json: input json file from tvr dataset
    :param img_dir: directory containing images
    :param subtitles: path to the json file containing subtitles
    :param output: output json file to store the results
    :return:
    """
    # load json file
    with open(input_json, "rt") as f:
        data = [json.loads(l) for l in f]
    # load subtitles
    with open(subtitles,"rt") as f:
        subs = [json.loads(l) for l in f]
    # convert subs to dictionary, indexed by vid_name
    subs = dict([(s["vid_name"],s["sub"]) for s in subs])
    print("number of videos in subtitles file:{0}".format(len(subs)))
    # load all image names
    clip_objs = list(filter(lambda x: x.is_dir() and x.name.find("clip")>-1,img_dir.glob("**/**")))
    # make sure 'clip' is found in the folder name
    clip_names = set([c.name for c in clip_objs])
    print("{0} clips found...".format(len(clip_names)))
    # filter missing clips
    moments = list(filter(lambda x: x["vid_name"] in clip_names,data))
    print("{0} moments after filtering missing data...".format(len(moments)))
    # num frames in each clip
    clip_num_frames = list(map(lambda x: len(list(x.glob("*.jpg"))), clip_objs))
    assert (len(clip_objs)==len(clip_num_frames)), "number of clip objects must equal num clip frames!"
    clip_name_to_num_frames = dict([(c.name,n) for c,n in zip(clip_objs, clip_num_frames)])
    clip_name_to_path = dict([(c.name,c.relative_to(img_dir).as_posix()) for c in clip_objs])
    # determine weights of label based on standard clip size
    max_num_frames = int(float(max(clip_name_to_num_frames.values())))
    print("max number of frames in all clips: {0}".format(max_num_frames))
    std_tgt = np.zeros([len(moments),max_num_frames]) # if all clips had the same length
    std_start = list(map(lambda m: round(m["ts"][0]/m["duration"]*max_num_frames), moments))
    std_end = list(map(lambda m: round(m["ts"][1]/m["duration"]*max_num_frames), moments))
    for ii,(ss,se) in enumerate(zip(std_start, std_end)):
        std_tgt[ii,ss:se+1] = 1
    std_num_ones = np.sum(std_tgt,axis=0)
    std_w_ones = 1. - std_num_ones / len(moments)
    # start/end frame based on timestamp
    results = dict()
    results["moments"] = []
    # append num_frames/start/end to each moment, save to output
    for m in moments:
        num_frames = clip_name_to_num_frames[m["vid_name"]]
        # start and end frame
        st = round(m["ts"][0]/m["duration"]*num_frames)
        ed = round(m["ts"][1]/m["duration"]*num_frames)
        m["num_frames"] = num_frames
        m["ts_frame"] = [st,ed]
        # subtitles
        # find intersecting subtitles with each moment
        m["subtitles"] = []
        for s in subs[m["vid_name"]]:
            if intersect([s["start"],s["end"]],m["ts"])>0:
                m["subtitles"].append(s)
        # relative path
        m["rel_path"] = clip_name_to_path[m["vid_name"]]
        # target label for each frame
        tgt = np.zeros(num_frames)
        tgt[st:ed+1] = 1
        m["frame_label"] = tgt.tolist()
        # weight for label=1 for frame label
        weights = [std_w_ones[int(round(float(ii)/num_frames*max_num_frames))] for ii in range(num_frames)]
        m["frame_pos_weights"] = weights
    # write to disk
    with open(output,"wt") as f:
        for m in moments:
            f.write(json.dumps(m)+"\n")

if __name__=="__main__":
    parser = create_parse()
    args = parser.parse_args()
    run(input_json=args.input_json, img_dir=args.img_dir, subtitles=args.subtitles, output=args.output)