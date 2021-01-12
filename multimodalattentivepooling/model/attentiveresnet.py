import torch.nn as nn
from .attentivepooling import CorpusAttentivePool
from .encoder import WordEncoder

from torchvision.models.utils import load_state_dict_from_url

__all__ = ['r3d_18', 'mc3_18', 'r2plus1d_18']

model_urls = {
    'r3d_18': 'https://download.pytorch.org/models/r3d_18-b3b3357e.pth',
    'mc3_18': 'https://download.pytorch.org/models/mc3_18-a90a0ba3.pth',
    'r2plus1d_18': 'https://download.pytorch.org/models/r2plus1d_18-91a641e6.pth',
}


class Conv3DSimple(nn.Conv3d):
    def __init__(self,
                 in_planes,
                 out_planes,
                 midplanes=None,
                 stride=1,
                 padding=1):

        super(Conv3DSimple, self).__init__(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=(3, 3, 3),
            stride=stride,
            padding=padding,
            bias=False)

    @staticmethod
    def get_downsample_stride(stride):
        return (stride, stride, stride)


class Conv2Plus1D(nn.Sequential):

    def __init__(self,
                 in_planes,
                 out_planes,
                 midplanes,
                 stride=1,
                 padding=1):
        super(Conv2Plus1D, self).__init__(
            nn.Conv3d(in_planes, midplanes, kernel_size=(1, 3, 3),
                      stride=(1, stride, stride), padding=(0, padding, padding),
                      bias=False),
            nn.BatchNorm3d(midplanes),
            nn.ReLU(inplace=True),
            nn.Conv3d(midplanes, out_planes, kernel_size=(3, 1, 1),
                      stride=(stride, 1, 1), padding=(padding, 0, 0),
                      bias=False))

    @staticmethod
    def get_downsample_stride(stride):
        return (stride, stride, stride)


class Conv3DNoTemporal(nn.Conv3d):

    def __init__(self,
                 in_planes,
                 out_planes,
                 midplanes=None,
                 stride=1,
                 padding=1):

        super(Conv3DNoTemporal, self).__init__(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=(1, 3, 3),
            stride=(1, stride, stride),
            padding=(0, padding, padding),
            bias=False)

    @staticmethod
    def get_downsample_stride(stride):
        return (1, stride, stride)


class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, inplanes, planes, conv_builder, stride=1, downsample=None, dw=128):
        midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)

        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(
            conv_builder(inplanes, planes, midplanes, stride),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
        )
        self.we1 = WordEncoder(d_model=dw, d_out=planes)
        self.pool1 = CorpusAttentivePool(kernel_size=(1,3,3), padding=(0,1,1))
        self.conv2 = nn.Sequential(
            conv_builder(planes, planes, midplanes),
            nn.BatchNorm3d(planes)
        )
        # self.we2 = WordEncoder(d_model=dw, d_out=planes)
        # self.pool2 = CorpusAttentivePool(kernel_size=(1,3,3), padding=(0,1,1))
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, vw_w):
        # print(type(vw_w),len(vw_w))
        vw, w = vw_w # unpack
        # vw: BCTHW
        # w: BTD
        residual = vw
        # word encoders
        wenc1 = self.we1(w)
        # wenc2 = self.we2(w)
        # print("word encodings:",wenc1.shape, wenc2.shape)
        # convert word encodings to shape accepted by pooling
        out = self.conv1(vw)
        # print("input to pool:",out.shape)
        out = self.pool1(out, wenc1)
        out = self.conv2(out)
        out = self.pool1(out, wenc1)
        # print("after pool2:",out.shape)
        if self.downsample is not None:
            residual = self.downsample(vw)

        out += residual
        out = self.relu(out)
        # print("returning:",out.shape)
        return [out, w] # this is to make sure next block in sequential works


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, conv_builder, stride=1, downsample=None):

        super(Bottleneck, self).__init__()
        midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)

        # 1x1x1
        self.conv1 = nn.Sequential(
            nn.Conv3d(inplanes, planes, kernel_size=1, bias=False),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
        )
        # Second kernel
        self.conv2 = nn.Sequential(
            conv_builder(planes, planes, midplanes, stride),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
        )

        # 1x1x1
        self.conv3 = nn.Sequential(
            nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm3d(planes * self.expansion)
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicStem(nn.Sequential):
    """The default conv-batchnorm-relu stem
    """
    def __init__(self):
        super(BasicStem, self).__init__(
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2),
                      padding=(1, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True))


class R2Plus1dStem(nn.Sequential):
    """R(2+1)D stem is different than the default one as it uses separated 3D convolution
    """
    def __init__(self):
        super(R2Plus1dStem, self).__init__(
            nn.Conv3d(3, 45, kernel_size=(1, 7, 7),
                      stride=(1, 2, 2), padding=(0, 3, 3),
                      bias=False),
            nn.BatchNorm3d(45),
            nn.ReLU(inplace=True),
            nn.Conv3d(45, 64, kernel_size=(3, 1, 1),
                      stride=(1, 1, 1), padding=(1, 0, 0),
                      bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True))


class VideoResNet(nn.Module):

    def __init__(self, block, conv_makers, layers, stem, dw, zero_init_residual=False):
        """Generic resnet video generator.

        Args:
            block (nn.Module): resnet building block
            conv_makers (list(functions)): generator function for each layer
            layers (List[int]): number of blocks per layer
            stem (nn.Module, optional): Resnet stem, if None, defaults to conv-bn-relu. Defaults to None.
            dw (int): Dimension of the input words encoding
            zero_init_residual (bool, optional): Zero init bottleneck residual BN. Defaults to False.
        """
        super(VideoResNet, self).__init__()
        self.inplanes = 64

        self.stem = stem()

        self.layer1 = self._make_layer(block, conv_makers[0], 64, layers[0], dw=dw, stride=1)
        self.layer2 = self._make_layer(block, conv_makers[1], 128, layers[1], dw=dw, stride=2)
        self.layer3 = self._make_layer(block, conv_makers[2], 256, layers[2], dw=dw, stride=2)
        self.layer4 = self._make_layer(block, conv_makers[3], 512, layers[3], dw=dw, stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.fc = nn.Linear(512 , 1) # yes/no classification

        # init weights
        self._initialize_weights()

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def forward(self, x, w):
        # expected shapes:
        # vw: BCTHW
        # w: BTD
        # print("input:",x.shape)
        x = self.stem(x)
        # print("stem:",x.shape)

        x = self.layer1([x, w])[0] # second element is w
        # print("l1:",x.shape)
        x = self.layer2([x, w])[0]
        # print("l2:",x.shape)
        x = self.layer3([x, w])[0]
        # print("l3:",x.shape)
        x = self.layer4([x, w])[0]
        # print("l4:",x.shape)

        x = self.avgpool(x)
        # remove HW singleton dims
        x = x.squeeze(3).squeeze(3).transpose(1,2) # BCT -> BTC, C is input to linear
        # CE expects input of type NC
        x = self.fc(x).squeeze(2) # output is BT
        # print("fc:",x.shape)

        return x

    def _make_layer(self, block, conv_builder, planes, blocks, dw=128, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            ds_stride = conv_builder.get_downsample_stride(stride)
            ds_stride = (1, ds_stride[1], ds_stride[2])
            # print("downsample stride:",ds_stride)
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=ds_stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion)
            )
        layers = []
        if stride==2:
            stride = (1, 2, 2)
        # print("layer stride:",stride)
        layers.append(block(self.inplanes, planes, conv_builder, stride, downsample, dw))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, conv_builder))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def _video_resnet(arch, pretrained=False, progress=True, **kwargs):
    model = VideoResNet(**kwargs)

    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def r3d_18(pretrained=False, progress=True, dw=300, **kwargs):
    """Construct 18 layer Resnet3D model as in
    https://arxiv.org/abs/1711.11248

    Args:
        pretrained (bool): If True, returns a model pre-trained on Kinetics-400
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        nn.Module: R3D-18 network
    """

    return _video_resnet('r3d_18',
                         pretrained, progress,
                         block=BasicBlock,
                         conv_makers=[Conv3DSimple] * 4,
                         layers=[2, 2, 2, 2],
                         stem=BasicStem,
                         dw=dw,
                         **kwargs)



def mc3_18(pretrained=False, progress=True, **kwargs):
    """Constructor for 18 layer Mixed Convolution network as in
    https://arxiv.org/abs/1711.11248

    Args:
        pretrained (bool): If True, returns a model pre-trained on Kinetics-400
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        nn.Module: MC3 Network definition
    """
    return _video_resnet('mc3_18',
                         pretrained, progress,
                         block=BasicBlock,
                         conv_makers=[Conv3DSimple] + [Conv3DNoTemporal] * 3,
                         layers=[2, 2, 2, 2],
                         stem=BasicStem, **kwargs)



def r2plus1d_18(pretrained=False, progress=True, **kwargs):
    """Constructor for the 18 layer deep R(2+1)D network as in
    https://arxiv.org/abs/1711.11248

    Args:
        pretrained (bool): If True, returns a model pre-trained on Kinetics-400
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        nn.Module: R(2+1)D-18 network
    """
    return _video_resnet('r2plus1d_18',
                         pretrained, progress,
                         block=BasicBlock,
                         conv_makers=[Conv2Plus1D] * 4,
                         layers=[2, 2, 2, 2],
                         stem=R2Plus1dStem, **kwargs)
