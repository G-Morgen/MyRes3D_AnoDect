'''ShuffleNetV2 in PyTorch.

See the paper "SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size" for more details.
'''

import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from functools import partial

__all__ = ['SqueezeNet', 'squeezenet1_0', 'squeezenet1_1']


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

class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes,
                 use_bypass=False):
        super(Fire, self).__init__()
        self.use_bypass = use_bypass
        self.inplanes = inplanes
        self.relu = nn.ReLU(inplace=True)
        self.squeeze = nn.Conv3d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_bn = nn.BatchNorm3d(squeeze_planes)
        self.expand1x1 = nn.Conv3d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_bn = nn.BatchNorm3d(expand1x1_planes)
        self.expand3x3 = nn.Conv3d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_bn = nn.BatchNorm3d(expand3x3_planes)

    def forward(self, x):
        out = self.squeeze(x)
        out = self.squeeze_bn(out)
        out = self.relu(out)

        out1 = self.expand1x1(out)
        out1 = self.expand1x1_bn(out1)
        
        out2 = self.expand3x3(out)
        out2 = self.expand3x3_bn(out2)

        out = torch.cat([out1, out2], 1)
        if self.use_bypass:
        	out += x
        out = self.relu(out)

        return out


class SqueezeNet(nn.Module):

    def __init__(self,
                 sample_size,
                 sample_duration,			
    	         version=1.1,
    	         num_classes=600):
        super(SqueezeNet, self).__init__()
        if version not in [1.0, 1.1]:
            raise ValueError("Unsupported SqueezeNet version {version}:"
                             "1.0 or 1.1 expected".format(version=version))
        self.num_classes = num_classes
        last_duration = int(math.ceil(sample_duration / 16))
        last_size = int(math.ceil(sample_size / 32))
        if version == 1.0:
            self.features = nn.Sequential(
                nn.Conv3d(3, 96, kernel_size=7, stride=(1,2,2), padding=(3,3,3)),
                nn.BatchNorm3d(96),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
                Fire(96, 16, 64, 64),
                Fire(128, 16, 64, 64, use_bypass=True),
                nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128, use_bypass=True),
                nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
                Fire(256, 48, 192, 192),
            )
            self.clf_features = nn.Sequential(
                Fire(384, 48, 192, 192, use_bypass=True),
                Fire(384, 64, 256, 256),
                nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
                Fire(512, 64, 256, 256, use_bypass=True),
                )
        if version == 1.1:
            self.features = nn.Sequential(
                nn.Conv3d(1, 64, kernel_size=3, stride=(1,2,2), padding=(1,1,1)),
                nn.BatchNorm3d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64, use_bypass=True),
                nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128, use_bypass=True),
                nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192, use_bypass=True),)
            self.clf_features = nn.Sequential(
                nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256, use_bypass=True),
            )
        # Final convolution is initialized differently form the rest
        final_conv = nn.Conv3d(512, self.num_classes, kernel_size=1)
        self.dec4    = DecoderBlock(384, 256, 256)
        self.dec3    = DecoderBlock(256, 128, 128)
        self.dec2    = DecoderBlock(128, 64, 64)
        self.dec1    = DecoderBlock(64, 32, 32, stride=(1,2,2))

        self.final = nn.Conv3d(32,1, kernel_size=1)
        self.final_layer = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AvgPool3d((last_duration, last_size, last_size), stride=1)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x, score=True):
        if not score:
            x = self.features(x)
            #print(x.size()) #torch.Size([8, 384, 4, 7, 7])
            x = self.dec4(x)
            x = self.dec3(x)
            x = self.dec2(x)
            x = self.dec1(x)
            x = self.final(x)
            return x

        else:
            x = self.features(x)
            #print (x.size())
            x = self.clf_features(x)
            #print (x.size())
            x = self.final_layer(x)
            return x.view(x.size(0), -1)


def get_fine_tuning_parameters(model, ft_potion):
    if ft_potion == "complete":
        return model.parameters()

    elif ft_potion == "last_layer":
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
    model = SqueezeNet(**kwargs)
    return model


if __name__ == '__main__':
    model = SqueezeNet(version=1.1, sample_size = 112, sample_duration = 16, num_classes=600)
    # model = model.cuda()
    # model = nn.DataParallel(model, device_ids=None)
    print(model)

    input_var = Variable(torch.randn(8, 3, 32, 112, 112))
    output = model(input_var, score=False)
    print(output.shape)