import argparse
from pathlib import Path
import numpy as np
from sklearn.metrics import confusion_matrix
from multimodalattentivepooling.utils.moduleload import load_comp, load_args
import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


def create_parse():
    parser = argparse.ArgumentParser(description='Prepare TVR dataset.')
    parser.add_argument('--dataset', '-ds', type=str, required=True, help='dataset module')
    parser.add_argument('--dataset-args', '-dsa', type=Path, required=True, help='dataset args YAML')
    parser.add_argument('--dataloader', '-dl', type=str, required=True, help='dataloader module')
    parser.add_argument('--dataloader-args', '-dla', type=Path, required=True, help='dataloader args YAML')
    parser.add_argument('--transforms', '-t', type=str, required=True, help='data transforms module')
    parser.add_argument('--transforms-args', '-tfa', type=Path, required=True, help='YAML file containing transforms args')
    parser.add_argument('--optimizer', '-o', type=str, required=True, help='optimizer')
    parser.add_argument('--optimizer-args', '-oa', type=Path, required=False, help='optimizer args YAML file')
    parser.add_argument('--net', '-n', type=str, required=True, help='network module')
    parser.add_argument('--net-args', '-na', type=Path, required=True, help='network args YAML')
    parser.add_argument('--loss', '-l', type=str, required=True, help='loss module')
    parser.add_argument('--loss-args', '-la', type=Path, required=False, help='loss args YAML')
    parser.add_argument('--train-args', '-ta', type=Path, required=True, help='train args YAML')
    return parser


def run(dataset, dataset_args, dataloader, dataloader_args, transforms, transforms_args,
                    optimizer, optimizer_args, net, net_args, loss, loss_args, train_args):
    """
    :param dataset (str): module specification of the dataset
    :param dataset_args: YAML file containing dataset arguments
    :param dataloader: module specification of the dataloader, for example torch.utils.data.DataLoader
    :param dataloader_args: YAML file containing dataloader arguments. The dataset will also be passed to this class.
    :param transforms: composite transforms class
    :param transforms_args: YAML file containing transforms arguments (pipeline of transform functions)
    :param optimizer: module for optimizer, e.g. torch.optim.Adam. network params will be pass to this class
    :param optimizer_args: YAML file containing optimizer params
    :param net: module specification for network
    :param net_args: YAML file specifying network arguments
    :param loss: loss module
    :param loss_args: YAML file containing loss arguments
    :param train_args: YAML file containing training args
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
    net, net_args = load_comp(net), load_args(net_args)
    net = net(**net_args)
    print("created network with {0} parameters".format(sum([np.prod(p.shape) for p in net.parameters() if p.requires_grad])))
    optimizer, optimizer_args = load_comp(optimizer), load_args(optimizer_args)
    optimizer = optimizer(net.parameters(),**optimizer_args)
    # scheduler
    scheduler = CosineAnnealingWarmRestarts(optimizer, 5000)
    loss, loss_args = load_comp(loss), load_args(loss_args)
    loss = loss(**loss_args)
    train_args = load_args(train_args)
    # training loop
    device = torch.device(train_args["device"]["name"])
    net.to(device)
    # net.eval()
    if "ckpt" in train_args.keys():
        print("loading from checkpoint:",device)
        net.load_state_dict(torch.load(train_args["ckpt"], map_location=device))
    loss.to(device)
    for epoch in range(train_args["nepochs"]):
        for epoch_index, data in enumerate(dataloader):
            optimizer.zero_grad()
            # pass data through transforms
            data = transforms(data)
            for n in train_args["device"]["data"]:
                data[n] = data[n].to(device)
            # network output
            data = net(data)
            # calculate loss, backpropagate, step
            data = loss(data)
            data["loss"].backward()
            optimizer.step()
            scheduler.step(epoch + epoch_index / len(dataloader))
            # misclassification
            if epoch_index%5==0:
                print(epoch, epoch_index,data["loss"].item())
                pred = data["win3"]
                # print(pred.shape)
                pred = torch.argmax(pred,dim=1).data.cpu().numpy().reshape(-1)
                gt = data["win3gt"].data.cpu().numpy().reshape(-1)
                # print(data["win33gt"].shape)
                idx = np.where(gt != -100)
                print(confusion_matrix(gt[idx], pred[idx]))
                print(data["sttgt_maxpooled"].tolist())
                print(torch.argmax(data["start_maxpooled"],dim=1).tolist())
                print(data["endtgt_maxpooled"].tolist())
                print(torch.argmax(data["end_maxpooled"],dim=1).tolist())
        torch.save(net.state_dict(), "/content/drive/MyDrive/modchunks.ckpt")



if __name__=="__main__":
    parser = create_parse()
    args = parser.parse_args()
    run(**vars(args))