'''ShuffleNet in PyTorch.

See the paper "ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable



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


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv3d(inp, oup, kernel_size=3, stride=stride, padding=(1,1,1), bias=False),
        nn.BatchNorm3d(oup),
        nn.ReLU(inplace=True)
    )


def channel_shuffle(x, groups):
    '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
    batchsize, num_channels, depth, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups, 
        channels_per_group, depth, height, width)
    #permute
    x = x.permute(0,2,1,3,4,5).contiguous()
    # flatten
    x = x.view(batchsize, num_channels, depth, height, width)
    return x



class Bottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, stride, groups):
        super(Bottleneck, self).__init__()
        self.stride = stride
        self.groups = groups
        mid_planes = out_planes//4
        if self.stride == 2:
            out_planes = out_planes - in_planes
        g = 1 if in_planes==24 else groups
        self.conv1    = nn.Conv3d(in_planes, mid_planes, kernel_size=1, groups=g, bias=False)
        self.bn1      = nn.BatchNorm3d(mid_planes)
        self.conv2    = nn.Conv3d(mid_planes, mid_planes, kernel_size=3, stride=stride, padding=1, groups=mid_planes, bias=False)
        self.bn2      = nn.BatchNorm3d(mid_planes)
        self.conv3    = nn.Conv3d(mid_planes, out_planes, kernel_size=1, groups=groups, bias=False)
        self.bn3      = nn.BatchNorm3d(out_planes)
        self.relu     = nn.ReLU(inplace=True)

        if stride == 2:
            self.shortcut = nn.AvgPool3d(kernel_size=(2,3,3), stride=2, padding=(0,1,1))


    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = channel_shuffle(out, self.groups)
        out = self.bn2(self.conv2(out))
        out = self.bn3(self.conv3(out))

        if self.stride == 2:
            out = self.relu(torch.cat([out, self.shortcut(x)], 1))
        else:
            out = self.relu(out + x)

        return out


class ShuffleNet(nn.Module):
    def __init__(self,
                 groups,
                 width_mult=1,
                 num_classes=400):
        super(ShuffleNet, self).__init__()
        self.num_classes = num_classes
        self.groups = groups
        num_blocks = [4,8,4]

        # index 0 is invalid and should never be called.
        # only used for indexing convenience.
        if groups == 1:
            out_planes = [24, 144, 288, 567]
        elif groups == 2:
            out_planes = [24, 200, 400, 800]
        elif groups == 3:
            out_planes = [24, 240, 480, 960]
        elif groups == 4:
            out_planes = [24, 272, 544, 1088]
        elif groups == 8:
            out_planes = [24, 384, 768, 1536]
        else:
            raise ValueError(
                """{} groups is not supported for
                   1x1 Grouped Convolutions""".format(num_groups))
        out_planes = [int(i * width_mult) for i in out_planes]
        self.in_planes = out_planes[0]
        self.conv1   = conv_bn(3, self.in_planes, stride=(1,2,2))
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1  = self._make_layer(out_planes[1], num_blocks[0], self.groups)
        self.layer2  = self._make_layer(out_planes[2], num_blocks[1], self.groups)

        self.dec4    = DecoderBlock(out_planes[2], 256, 256)
        self.dec3    = DecoderBlock(256, 128, 128)
        self.dec2    = DecoderBlock(128, 64, 64)
        self.dec1    = DecoderBlock(64, 32, 32, stride=(1,2,2))

        self.final = nn.Conv3d(32,2, kernel_size=1)
        self.sigmoid = nn.Sigmoid()


        # building classifier: This part is only getting anomaly score!
        self.layer3  = self._make_layer(out_planes[3], num_blocks[2], self.groups)
        self.final_linear = nn.Sequential(
                        nn.Dropout(0.2),
                        nn.Linear(out_planes[3], self.num_classes)
                        )

    def _make_layer(self, out_planes, num_blocks, groups):
        layers = []
        for i in range(num_blocks):
            stride = 2 if i == 0 else 1
            layers.append(Bottleneck(self.in_planes, out_planes, stride=stride, groups=groups))
            self.in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x, score):
        if not score: # autoencoder for normal driving
            #print ("input:",x.size()) # input: torch.Size([16, 2, 32, 112, 112])
            out = self.conv1(x)
            #print ("after conv1:",out.size()) #after conv1: torch.Size([16, 24, 32, 56, 56])
            out = self.maxpool(out)
            #print ("after mp:",out.size()) #after mp: torch.Size([16, 24, 16, 28, 28])
            out = self.layer1(out)
            #print ("after layer1:",out.size()) #after layer1: torch.Size([16, 240, 8, 14, 14])
            out = self.layer2(out) 
            #print ("after layer2:",out.size()) #after layer2: torch.Size([16, 480, 4, 7, 7])
            out = self.dec4(out)
            #print ("after dec4:",out.size()) #after dec4: torch.Size([16, 256, 8, 14, 14])
            out = self.dec3(out)
            #out = self.dec3(torch.cat((out,out1),dim=1))
            #print ("after dec3:",out.size()) #after dec3: torch.Size([16, 128, 16, 28, 28])
            out = self.dec2(out)
            #print ("after dec2:",out.size()) #after dec2: torch.Size([16, 64, 32, 56, 56])
            out = self.dec1(out)
            #print ("after dec1:",out.size()) #after dec1: torch.Size([16, 32, 32, 112, 112])
            out = self.final(out) # removed the sigmoid
            #print ("after final:",out.size()) #after final: torch.Size([16, 2, 32, 112, 112])
            # out = self.sigmoid(out)
            return out
        else:
            out = self.conv1(x)
            out = self.maxpool(out)
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = F.avg_pool3d(out, out.data.size()[-3:])
            out = out.view(out.size(0), -1)
            out = self.final_linear(out)
            return out

def get_fine_tuning_parameters(model, ft_portion):
    if ft_portion == "complete":
        return model.parameters()

    elif ft_portion == "last_layer":
        ft_module_names = []
        ft_module_names.append('classifier')

        parameters = []
        for k, v in model.named_parameters():
            for ft_module in ft_module_names:
                if ft_module in k:
                    parameters.append({'params': v})
                    break
            else:
                parameters.append({'params': v, 'lr': 0.0})
        return parameters

    else:
        raise ValueError("Unsupported ft_potion: 'complete' or 'last_layer' expected")


def get_model(**kwargs):
    """
    Returns the model.
    """
    model = ShuffleNet(**kwargs)
    return model


if __name__ == "__main__":
    model = get_model(groups=3, num_classes=2, width_mult=1)
    model = model.cuda()
    model = nn.DataParallel(model, device_ids=None)
    print(model)

    input_var = Variable(torch.randn(8, 1, 16, 112, 112))
    output = model(input_var, score=True)
    print(output.shape)
    flops, prms = profile(model, input_size=(1, 1, 32, 112, 112))
    print("Total number of FLOPs: ", flops)
    print("Total number of prms: ", prms)


