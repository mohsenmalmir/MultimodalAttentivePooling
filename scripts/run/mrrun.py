import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from skimage import transform
import gensim.downloader
from gensim.parsing.preprocessing import remove_stopwords, preprocess_string
import sys
sys.path.append('/home/mmalmir/MultimodalAttentivePooling')
from multimodalattentivepooling.dataset.momentretrieval import MomentRetrieval
from multimodalattentivepooling.model.attentiveresnet import r3d_18
from torch.utils.tensorboard import SummaryWriter

# using global word2vec model
glove_vectors = gensim.downloader.load('glove-wiki-gigaword-300')

### transformation functions:
def word2vec(data):
    """
    given a sample data from dataset object, this function will transform the
    query words to features using word2vec
    """
    q = data["query"]
    # remove stop words
    q = remove_stopwords(q)
    # prepare string
    q = preprocess_string(q)
    # word2vec

    enc = [glove_vectors[w].reshape([1,-1]) if w in glove_vectors else np.zeros([1,300]) for w in q]
    data["query_encoding"] = np.concatenate(enc,axis=0)
    return data

def imresize(data):
    """
    resize all images in the clip, the concatenate them into one tensor
    """
    imgs = data["image"]
    h, w = imgs[0].shape[:2]
    output_size = 101
    if h < w:
        new_h, new_w = output_size * h / w, output_size
    else:
        new_h, new_w = output_size, output_size * w / h

    new_h, new_w = int(new_h), int(new_w)

    imgs = list(map(lambda x: np.transpose(transform.resize(x, (new_h, new_w)),[2,0,1]), imgs))
    imgs = np.concatenate([img[:,np.newaxis,...] for img in imgs],axis=1)
    data["image"] = imgs
    return data

def labelprep(data):
    data["frame_label"] = np.asarray(data["frame_label"])
    data["frame_label_pos_weights"] = np.asarray(data["frame_label_pos_weights"])
    return data

# data loader
print("creating the dataloader...")
tds = MomentRetrieval(Path("/home/mmalmir/tvr_train.json"),Path("/home/mmalmir/frames_hq/"),transform=[word2vec,imresize,labelprep])
trainloader = DataLoader(tds, batch_size=1, shuffle=True, num_workers=4)

for ds in trainloader:
    print(ds["image"].shape)
    print(ds["frame_label"].shape)
    print(ds["frame_label_pos_weights"].shape)
    break

# creating the network
print("creating the network...")
net = r3d_18(dw=300)

print("creating the optimizer...")
optimizer = optim.Adam(net.parameters())

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device in use:",device)
net = net.to(device)

print("entering the training loop...")
n_epochs = 100
writer = SummaryWriter()
cntr = 0
for epoch in range(n_epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        cntr += 1
        # prep input
        imgs, Q, L, W = data["image"], data["query_encoding"], data["frame_label"], data["frame_label_pos_weights"]
        imgs, Q, W = list(map(lambda x: x.float(), [imgs, Q, W]))
        # imgs, L, W = imgs[:,:,::4,:,:], L[:,::4], W[:,::4]
        L = L.float()
        imgs, Q, L, W = list(map(lambda x: x.to(device), [imgs, Q, L, W]))
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(imgs, Q)
        loss = F.binary_cross_entropy_with_logits(outputs, L, pos_weight=W)
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
        writer.add_scalar("Loss/train", loss, cntr)
        if i % 20 == 19:    # print every 2000 mini-batches
            print('[%d, %5d] loss:%.3f running loss: %.3f' %
                  (epoch + 1,  cntr, loss.item(), running_loss / 20))
            running_loss = 0.0

        if i%1000==999:
            torch.save({
                'epoch': EPOCH,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': LOSS,
            }, "./model_checkpt_{0}".format(i))
