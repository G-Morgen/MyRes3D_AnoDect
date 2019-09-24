'''ShuffleNetV2 in PyTorch.

See the paper "ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design" for more details.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init
import math


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv3d(inp, oup, kernel_size=3, stride=stride, padding=(1,1,1), bias=False),
        nn.BatchNorm3d(oup),
        nn.ReLU(inplace=True)
    )

def conv_1x1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv3d(inp, oup, 1, 1, 0, bias=False),
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
    
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        oup_inc = oup//2
        
        if self.stride == 1:
            #assert inp == oup_inc
        	self.banch2 = nn.Sequential(
                # pw
                nn.Conv3d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup_inc),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv3d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
                nn.BatchNorm3d(oup_inc),
                # pw-linear
                nn.Conv3d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup_inc),
                nn.ReLU(inplace=True),
            )                
        else:                  
            self.banch1 = nn.Sequential(
                # dw
                nn.Conv3d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm3d(inp),
                # pw-linear
                nn.Conv3d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup_inc),
                nn.ReLU(inplace=True),
            )        
    
            self.banch2 = nn.Sequential(
                # pw
                nn.Conv3d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup_inc),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv3d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
                nn.BatchNorm3d(oup_inc),
                # pw-linear
                nn.Conv3d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup_inc),
                nn.ReLU(inplace=True),
            )
          
    @staticmethod
    def _concat(x, out):
        # concatenate along channel axis
        return torch.cat((x, out), 1)        

    def forward(self, x):
        if self.stride == 1:
            x1 = x[:, :(x.shape[1]//2), :, :, :]
            x2 = x[:, (x.shape[1]//2):, :, :, :]
            out = self._concat(x1, self.banch2(x2))
        elif self.stride == 2:
            out = self._concat(self.banch1(x), self.banch2(x))

        return channel_shuffle(out, 2)


class ShuffleNetV2(nn.Module):
    def __init__(self, num_classes=600, sample_size=112, width_mult=1.):
        super(ShuffleNetV2, self).__init__()
        assert sample_size % 16 == 0
        
        self.stage_repeats = [4, 8]
        # index 0 is invalid and should never be called.
        # only used for indexing convenience.
        if width_mult == 0.25:
            self.stage_out_channels = [-1, 24,  32,  64, 128, 1024]
        elif width_mult == 0.5:
            self.stage_out_channels = [-1, 24,  48,  96, 192, 1024]
        elif width_mult == 1.0:
            self.stage_out_channels = [-1, 24, 116, 232, 464, 1024]
        elif width_mult == 1.5:
            self.stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        elif width_mult == 2.0:
            self.stage_out_channels = [-1, 24, 224, 488, 976, 2048]
        else:
            raise ValueError(
                """{} groups is not supported for
                       1x1 Grouped Convolutions""".format(num_groups))

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.conv1 = conv_bn(2, input_channel, stride=(1,2,2))
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        self.features = []
        # building inverted residual blocks
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage+2]
            for i in range(numrepeat):
                stride = 2 if i == 0 else 1
                self.features.append(InvertedResidual(input_channel, output_channel, stride))
                input_channel = output_channel
                
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        self.dec4    = DecoderBlock(output_channel, 256, 256)
        self.dec3    = DecoderBlock(256, 128, 128)
        self.dec2    = DecoderBlock(128, 64, 64)
        self.dec1    = DecoderBlock(64, 32, 32, stride=(1,2,2))
        self.final = nn.Conv3d(32,2, kernel_size=1)

        self.clf_stage_repeats = [4]
        self.clf_features = []
        # building inverted residual blocks
        for idxstage in range(len(self.clf_stage_repeats)):
            numrepeat = self.clf_stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage+4]
            for i in range(numrepeat):
                stride = 2 if i == 0 else 1
                self.clf_features.append(InvertedResidual(input_channel, output_channel, stride))
                input_channel = output_channel
        self.clf_features = nn.Sequential(*self.clf_features)

        # building last several layers
        self.conv_last      = conv_1x1x1_bn(input_channel, self.stage_out_channels[-1])
    
	    # building classifier
        self.final_layer = nn.Sequential(
                            nn.Dropout(0.2),
                            nn.Linear(self.stage_out_channels[-1], num_classes)
                            )

    def forward(self, x, score):
        # print ("1:",x.size())
        if not score:
            out = self.conv1(x)
            #print (out.size()) #torch.Size([8, 24, 32, 56, 56])
            out = self.maxpool(out)
            # print (out.size()) #torch.Size([8, 24, 16, 28, 28])
            out = self.features(out)
            # print (out.size()) #torch.Size([8, 464, 2, 4, 4])
            out = self.dec4(out)
            # print ("after dec4:",out.size()) #after dec4: torch.Size([16, 256, 8, 14, 14])
            out = self.dec3(out)
            #out = self.dec3(torch.cat((out,out1),dim=1))
            # print ("after dec3:",out.size()) #after dec3: torch.Size([16, 128, 16, 28, 28])
            out = self.dec2(out)
            # print ("after dec2:",out.size()) #after dec2: torch.Size([16, 64, 32, 56, 56])
            out = self.dec1(out)
            # print ("after dec1:",out.size()) #after dec1: torch.Size([16, 32, 32, 112, 112])
            out = self.final(out)
            # print ("2:",out.size())
            return out

        else:    
            out = self.conv1(x)
            #print (out.size()) #torch.Size([8, 24, 32, 56, 56])
            out = self.maxpool(out)
            #print (out.size()) #torch.Size([8, 24, 16, 28, 28])
            out = self.features(out)
            #print (out.size()) #torch.Size([8, 464, 2, 4, 4])
            out = self.clf_features(out)
            #print (out.size()) #torch.Size([8, 232, 4, 7, 7])
            out = self.conv_last(out)
            #print (out.size()) #torch.Size([8, 1024, 2, 4, 4])
            out = F.avg_pool3d(out, out.data.size()[-3:])
            #print (out.size()) #torch.Size([8, 1024, 1, 1, 1])
            out = out.view(out.size(0), -1)
            #print (out.size())
            out = self.final_layer(out)
            return out


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
    model = ShuffleNetV2(**kwargs)
    return model
   

if __name__ == "__main__":
    model = get_model(num_classes=2, sample_size=112, width_mult=1.)
    # model = model.cuda()
    # model = nn.DataParallel(model, device_ids=None)
    print(model)

    input_var = Variable(torch.randn(8, 3, 32, 112, 112))
    output = model(input_var,score=False)
    print(output.shape)