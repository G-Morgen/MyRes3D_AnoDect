import torch
from torch import nn

from models import c3d, squeezenet, mobilenet, shufflenet, mobilenetv2, shufflenetv2


def generate_model(opt):
    assert opt.model in ['c3d', 'squeezenet', 'mobilenet', 
                         'shufflenet', 'mobilenetv2', 'shufflenetv2']


    if opt.model == 'c3d':
        from models.c3d import get_fine_tuning_parameters
        model = c3d.get_model(
            num_classes=opt.n_classes,
            sample_size=opt.sample_size,
            sample_duration=opt.sample_duration)
    elif opt.model == 'squeezenet':
        from models.squeezenet import get_fine_tuning_parameters
        model = squeezenet.get_model(
            version=opt.version,
            num_classes=opt.n_classes,
            sample_size=opt.sample_size,
            sample_duration=opt.sample_duration)
    elif opt.model == 'shufflenet':
        from models.shufflenet import get_fine_tuning_parameters
        model = shufflenet.get_model(
            groups=opt.groups,
            width_mult=opt.width_mult,
            num_classes=opt.n_classes)
    elif opt.model == 'shufflenetv2':
        from models.shufflenetv2 import get_fine_tuning_parameters
        model = shufflenetv2.get_model(
            num_classes=opt.n_classes,
            sample_size=opt.sample_size,
            width_mult=opt.width_mult)
    elif opt.model == 'mobilenet':
        from models.mobilenet import get_fine_tuning_parameters
        model = mobilenet.get_model(
            num_classes=opt.n_classes,
            sample_size=opt.sample_size,
            width_mult=opt.width_mult)
    elif opt.model == 'mobilenetv2':
        from models.mobilenetv2 import get_fine_tuning_parameters
        model = mobilenetv2.get_model(
            num_classes=opt.n_classes,
            sample_size=opt.sample_size,
            width_mult=opt.width_mult)



    if not opt.no_cuda:
        model = model.cuda()
        model = nn.DataParallel(model, device_ids=None)
        pytorch_total_params = sum(p.numel() for p in model.parameters() if
                               p.requires_grad)
        print("Total number of trainable parameters: ", pytorch_total_params)

        if opt.pretrain_path:
            print('loading pretrained model {}'.format(opt.pretrain_path))
            pretrain = torch.load(opt.pretrain_path, map_location=torch.device('cpu'))
            model_dict = model.state_dict()
            pretrain_dict = {k: v for k, v in pretrain['state_dict'].items() if k in model_dict} 
            model_dict.update(pretrain_dict)
            model.load_state_dict(model_dict)

            # if opt.model in  ['mobilenet', 'mobilenetv2', 'shufflenet', 'shufflenetv2']:
            #     model.module.classifier = nn.Sequential(
            #                     nn.Dropout(0.8),
            #                     nn.Linear(model.module.classifier[1].in_features, opt.n_finetune_classes))
            #     model.module.classifier = model.module.classifier.cuda()
            # elif opt.model == 'squeezenet':
            #     model.module.classifier = nn.Sequential(
            #                     nn.Dropout(p=0.5),
            #                     nn.Conv3d(model.module.classifier[1].in_channels, opt.n_finetune_classes, kernel_size=1),
            #                     nn.ReLU(inplace=True),
            #                     nn.AvgPool3d((1,4,4), stride=1))
            #     model.module.classifier = model.module.classifier.cuda()
            # else:
            #     model.module.fc = nn.Linear(model.module.fc.in_features, opt.n_finetune_classes)
            #     model.module.fc = model.module.fc.cuda()

            model = _modify_first_conv_layer(model)
            model = model.cuda()
            
            parameters = get_fine_tuning_parameters(model, opt.ft_portion)
            return model, parameters
    else:
        if opt.pretrain_path:
            print('loading pretrained model {}'.format(opt.pretrain_path))
            pretrain = torch.load(opt.pretrain_path)
            assert opt.arch == pretrain['arch']
            model.load_state_dict(pretrain['state_dict'])

            if opt.model in  ['mobilenet', 'mobilenetv2', 'shufflenet', 'shufflenetv2']:
                model.module.classifier = nn.Sequential(
                                nn.Dropout(0.9),
                                nn.Linear(model.module.classifier[1].in_features, opt.n_finetune_classes)
                                )
            elif opt.model == 'squeezenet':
                model.module.classifier = nn.Sequential(
                                nn.Dropout(p=0.5),
                                nn.Conv3d(model.module.classifier[1].in_channels, opt.n_finetune_classes, kernel_size=1),
                                nn.ReLU(inplace=True),
                                nn.AvgPool3d((1,4,4), stride=1))
            else:
                model.module.fc = nn.Linear(model.module.fc.in_features, opt.n_finetune_classes)

            parameters = get_fine_tuning_parameters(model, opt.ft_begin_index)
            return model, parameters

    return model, model.parameters()


# def _modify_first_conv_layer(base_model):
#     modules = list(base_model.modules())
#     first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv3d),
#                                  list(range(len(modules)))))[0]
#     conv_layer = modules[first_conv_idx]
#     container = modules[first_conv_idx - 1]

#     new_conv = nn.Conv3d(2, conv_layer.out_channels, kernel_size=3,
#                          stride=(1,2,2), padding=1, bias=False)
#     layer_name = list(container.state_dict().keys())[0][:-7]

#     setattr(container, layer_name, new_conv)
#     return base_model


def _modify_first_conv_layer(model):
    modules = list(model.modules())
    first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv3d),
                                 list(range(len(modules)))))[0]
    conv_layer = modules[first_conv_idx]
    container = modules[first_conv_idx - 1] # container is in [first_conv_idx - 3] apparently.

    # modify parameters, assume the first blob contains the convolution kernels
    params = [x.clone() for x in conv_layer.parameters()]
    kernel_size = params[0].size()
    new_kernel_size = kernel_size[:1] + (2, ) + kernel_size[2:]
    new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()

    new_conv = nn.Conv3d(1, conv_layer.out_channels,
                         conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                         bias=True if len(params) == 2 else False)
    new_conv.weight.data = new_kernels
    if len(params) == 2:
        new_conv.bias.data = params[1].data # add bias if neccessary

    layer_name = list(container.state_dict().keys())[0][:-7]
    setattr(container, layer_name, new_conv)

    return model