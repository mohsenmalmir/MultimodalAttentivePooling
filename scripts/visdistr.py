import json
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS


def plot_grid(x_ticks, y_ticks, grid, cell_size):
    NX, NY = len(x_ticks)+1, len(y_ticks)+1
    H, W = cell_size
    img = np.ones([NY*W,NX*H,3])*255
    for ii in range(NX-1):
        for jj in range(NY-1):
            cx, cy = (ii+1)*H + H//2, (NY-jj-2)*W+W//2
            cv2.circle(img, (cx, cy), grid[ii-1, jj-1], (0,255,0), -1)
    for ii in range(NX-1):
        cx, cy = (ii+1)*H,(NY)*W
        cv2.putText(img,x_ticks[ii],(cx,cy),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2,cv2.LINE_AA)
    for jj in range(NY-1):
        cx, cy = 0,(NY-jj-2)*W+W//2
        cv2.putText(img,y_ticks[jj],(cx,cy),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2,cv2.LINE_AA)
    return img

def plot(x_data, y_data):
    percentiles = [10,20,30,40,50,60,70,80,90,100]
    X = np.percentile(x_data,percentiles)
    X = np.unique(X)
    Y = np.percentile(y_data,percentiles)
    Y = np.unique(Y)
    NX, NY = X.shape[0], Y.shape[0]
    x_ticks = ["%2.2f"%x for x in X]
    y_ticks = ["%2.2f"%y for y in Y]
    grid = np.zeros([len(X),len(Y)],dtype=int)
    for ii in range(len(x_data)):
        xx, yy = x_data[ii], y_data[ii]
        xidx = np.where(X>=xx)[0][0] # largest percentile that is smaller than start
        yidx = np.where(Y>=yy)[0][0]
        grid[xidx,yidx] = grid[xidx,yidx]+1
    grid = (grid.astype(float) / 10).astype(int)
    # print(X,Y,grid)
    img = plot_grid(x_ticks,y_ticks,grid,(100,100))
    cv2.imshow("grid",img)
    cv2.waitKey(0)


def create_parse():
    parser = argparse.ArgumentParser(description='Prepare TVR dataset.')
    parser.add_argument('--pred', '-p', type=str, required=True, help='prediction json file')
    parser.add_argument('--gt', '-g', type=str, required=True, help='ground truth json file')
    return parser

def run(pred,gt):
    with open(pred,"r") as f:
        pred = json.load(f)
    with open(gt,"r") as f:
        gt = [json.loads(l) for l in f]
    # parse pred
    vid2idx = pred["video2idx"]
    print(len(vid2idx.keys()))
    pred = pred["SVMR"]
    print(len(pred),len(gt))
    print(pred[-1])
    print(gt[0])
    did2ts_gt = dict([(g['desc_id'],g['ts']) for g in gt])
    did2desc_gt = dict([(g['desc_id'],g['desc']) for g in gt])
    did2ts_pred = dict([(p['desc_id'],p['predictions'][0][1:3]) for p in pred])
    did2dur_gt = dict([(g['desc_id'],g['duration']) for g in gt])
    did2vidx_pred = dict([(p['desc_id'],p['predictions'][0][0]) for p in pred])
    did2vidx_gt = dict([(g['desc_id'],vid2idx[g['vid_name']]) for g in gt])
    assert(len(set(did2ts_gt.keys()).intersection(did2ts_pred.keys())) == len(did2ts_pred.keys()))
    assert(all([did2vidx_pred[k]==did2vidx_gt[k] for k in did2vidx_gt.keys()]))
    # gt moments grid, X=moment start, Y=moment length, normalized to video len
    moment_len_gt, moment_start_gt = [], []
    moment_len_pred, moment_start_pred = [], []
    ious, descs = [], []
    for jj,k in enumerate(did2ts_gt.keys()):
        vid_dur = did2dur_gt[k]
        s_gt, e_gt = did2ts_gt[k]
        s_pred, e_pred = did2ts_pred[k]
        ms_gt, len_gt = s_gt/vid_dur, (e_gt-s_gt)/vid_dur
        ms_pred, len_pred = s_pred/vid_dur, (e_pred-s_pred)/vid_dur
        moment_len_gt.append(len_gt)
        moment_start_gt.append(ms_gt)
        moment_len_pred.append(len_pred)
        moment_start_pred.append(ms_pred)
        ious.append(max(0,min(e_gt,e_pred)-max(s_gt,s_pred))/(max(e_gt,e_pred)-min(s_gt,s_pred)))
        descs.append(did2desc_gt[k])
        # if jj>100:
        #     break
    plt.hist(ious,bins=100,density=True)
    plt.show()
    # print(ious)
    plot(moment_start_gt, moment_len_gt)
    plot(moment_start_pred, moment_len_pred)
    plot(moment_start_gt, ious)
    plot(moment_start_pred, ious)
    plot(sorted(ious),[jj/len(ious) for jj in range(len(ious))])
    idx = np.argsort(ious)
    for jj,ii in enumerate(idx):
        print(jj/len(ious),ious[ii],descs[ii])
    text = " ".join([descs[ii] for ii in range(len(ious)) if ious[ii]<0.5])
    wordcloud = WordCloud(width = 3000, height = 2000, random_state=1, background_color='blue',
                          collocations=False, stopwords = STOPWORDS).generate(text)
    plt.figure(figsize=(40, 30))
    # Display image
    plt.imshow(wordcloud)
    # No axis
    plt.axis("off")
    # plt.savefig("wordcloud.jpg")
    plt.show()

if __name__=="__main__":
    parser = create_parse()
    args = parser.parse_args()
    run(**vars(args))