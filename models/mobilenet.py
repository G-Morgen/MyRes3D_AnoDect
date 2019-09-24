'''MobileNet in PyTorch.

See the paper "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv3d(inp, oup, kernel_size=3, stride=stride, padding=(1,1,1), bias=False),
        nn.BatchNorm3d(oup),
        nn.ReLU(inplace=True)
    )

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, stride=None):
        super().__init__()
        if stride:
            s = stride
            p = (0,1,1)
        else:
            s = (2,2,2)
            p = (1,1,1)
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, middle_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(middle_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(middle_channels, out_channels, kernel_size=3, stride=s, padding=1, output_padding=p),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm3d(in_planes)
        self.conv2 = nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm3d(out_planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out


class MobileNet(nn.Module):
    def __init__(self, num_classes=600, sample_size=224, width_mult=1.):
        super(MobileNet, self).__init__()

        input_channel = 32
        last_channel = 1024
        input_channel = int(input_channel * width_mult)
        last_channel = int(last_channel * width_mult)
        cfg = [
        # c, n, s
        [64,   1, (2,2,2)],
        [128,  2, (2,2,2)],
        [256,  2, (2,2,2)],
        ]

        cfg_clf = [
        [512,  6, (2,2,2)],
        [1024, 2, (1,1,1)],
        ]

        self.features = [conv_bn(1, input_channel, (1,2,2))]
        # building inverted residual blocks
        for c, n, s in cfg:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(Block(input_channel, output_channel, stride))
                input_channel = output_channel
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        self.dec4    = DecoderBlock(output_channel, 256, 256)
        self.dec3    = DecoderBlock(256, 128, 128)
        self.dec2    = DecoderBlock(128, 64, 64)
        self.dec1    = DecoderBlock(64, 32, 32, stride=(1,2,2))

        self.final = nn.Conv3d(32,1, kernel_size=1)

        self.clf_features = []
        for c, n, s in cfg_clf:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i ==0 else 1
                self.clf_features.append(Block(input_channel, output_channel, stride))
                input_channel = output_channel
        self.clf_features = nn.Sequential(*self.clf_features)

        # building classifier
        self.final_layer = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes),
        )


    def forward(self, x,score):
        if not score:
            x = self.features(x)
            # print (x.size())
            x = self.dec4(x)
            x = self.dec3(x)
            x = self.dec2(x)
            x = self.dec1(x)
            x = self.final(x)
            return x

        else:
            x = self.features(x)
            # print (x.size())
            x = self.clf_features(x)
            # print (x.size())
            x = F.avg_pool3d(x, x.data.size()[-3:])
            # print (x.size())
            x = x.view(x.size(0), -1)
            x = self.final_layer(x)
            return x


def get_fine_tuning_parameters(model, ft_begin_index):
    ft_begin_index=0 if ft_begin_index=="complete" else ft_begin_index
    if ft_begin_index == 0:
        return model.parameters()

    ft_module_names = []
    for i in range(ft_begin_index, 5):
        ft_module_names.append('layer{}'.format(i))
    ft_module_names.append('classifier')

    parameters = []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                parameters.append({'params': v})
                break
        else:
            parameters.append({'params': v, 'lr': 0.0})
            #parameters.append({'params': v, 'requires_grad': False})

    return parameters
    

def get_model(**kwargs):
    """
    Returns the model.
    """
    model = MobileNet(**kwargs)
    return model



if __name__ == '__main__':
    model = get_model(num_classes=2, sample_size = 112, width_mult=1.)
    # model = model.cuda()
    # model = nn.DataParallel(model, device_ids=None)
    print(model)

    input_var = Variable(torch.randn(8, 3, 32, 112, 112))
    output = model(input_var, score=True)
    print(output.shape)
