# from model.i3d import I3D
# from model.r2p1d import R2Plus1DNet
# from model.r3d import resnet18, resnet34, resnet50
# from model.c3d import C3D
# from model.s3d_g import S3D_G
# from model.s3d import S3DG
import torch.nn as nn
from model.model import TCN, Flatten, Normalize
import torch
from utils.load_weights import ft_load_weight
import models

def pt_model_config(args, num_class):
    """

    :param args:
    :param num_class: num_class is only use for supervised train
    :return:
    """
    if args.arch == 'i3d':
        pass
        # model = I3D(num_classes=101, modality=args.pt_mode, with_classifier=False)
        # model_ema = I3D(num_classes=101, modality=args.pt_mode,  with_classifier=False)
    elif args.arch == 'r2p1d':
        model = R2Plus1DNet((1, 1, 1, 1), num_classes=num_class, with_classifier=False)
        model_ema = R2Plus1DNet((1, 1, 1, 1), num_classes=num_class, with_classifier=False)

    elif args.arch == "r2p1d_lateTemporal":
        tmp = R2Plus1DNet((1, 1, 1, 1), num_classes=num_class, with_classifier=False)
        model =  models.__dict__[args.ssl_arch](tmp=tmp, modelPath='')
        # model.id_head = tmp.id_head
        model_ema = models.__dict__[args.ssl_arch](tmp=tmp, modelPath='')
        
        if args.ssl_load_model:
            #Load state_dict
            params = torch.load(args.ssl_checkpoint)
            save_model = params['state_dict_encoder']

            # print(params.keys())

            # keys = save_model.keys()
            # print(keys)
        
            model_dict =  model.state_dict()
            state_dict = {k:v for k,v in save_model.items() if k in model_dict.keys()}
            # print(state_dict.keys())  # dict_keys(['w', 'conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias'])
            model_dict.update(state_dict)
            model.load_state_dict(model_dict)
            model_ema.load_state_dict(model_dict)
        
        model = torch.nn.DataParallel(model)
        model_ema = torch.nn.DataParallel(model_ema)

        return model, model_ema, params

        # model_ema.id_head = tmp.id_head
    # elif args.arch == 'r3d18':
    #     model = resnet18(num_classes=num_class, with_classifier=False)
    #     model_ema = resnet18(num_classes=num_class, with_classifier=False)
    # elif args.arch == 'r3d34':
    #     model = resnet34(num_classes=num_class, with_classifier=False)
    #     model_ema = resnet34(num_classes=num_class, with_classifier=False)
    # elif args.arch == 'r3d50':
    #     model = resnet50(num_classes=num_class, with_classifier=False)
    #     model_ema = resnet50(num_classes=num_class, with_classifier=False)
    # elif args.arch == 'c3d':
    #     model = C3D(with_classifier=False, num_classes=num_class)
    #     model_ema = C3D(with_classifier=False, num_classes=num_class)
    # elif args.arch == 's3d':
    #     model = S3D_G(num_class=num_class, in_channel=3, gate=True, with_classifier=False)
    #     model_ema = S3D_G(num_class=num_class, in_channel=3, gate=True, with_classifier=False)
    else:
        Exception("Not implemene error!")
    model = torch.nn.DataParallel(model)
    model_ema = torch.nn.DataParallel(model_ema)
    return model, model_ema

class SSLModel(nn.Module):
    def __init__(self, args):
        super(SSLModel, self).__init__()
        pretrained = (not args.train_from_scratch)
        self.encoder =  models.__dict__[args.ssl_arch](freeze_backbone=args.freeze_backbone, pretrained=pretrained, modelPath='')
        self.id_head = nn.Sequential(
                torch.nn.AdaptiveAvgPool3d((1, 1, 1)),
                Flatten(),
                torch.nn.Linear(512, 128),
                Normalize(2)
            )
    def forward(self, x):
        out = self.encoder(x)
        out = self.id_head(out)
        return out

def pt_model_config_ddp(args, num_class):
    """

    :param args:
    :param num_class: num_class is only use for supervised train
    :return:
    """
    if args.arch == "r2p1d_lateTemporal":
        model =  SSLModel(args)
        model_ema = SSLModel(args)
    else:
        Exception("Not implemented error!")
    return model, model_ema

def ft_model_config(args, num_class):
    with_classifier = True
    if args.arch == 'i3d':
        base_model = I3D(num_classes=num_class, modality=args.ft_mode, dropout_prob=args.ft_dropout, with_classifier=with_classifier)
        # args.logits_channel = 1024
        if args.ft_spatial_size == '112':
            out_size = (int(args.ft_data_length) // 8, 4, 4)
        else:
            out_size = (int(args.ft_data_length) // 8, 7, 7)
    elif args.arch == 'r2p1d':
        base_model = R2Plus1DNet((1, 1, 1, 1), num_classes=num_class, with_classifier=with_classifier)
        # args.logits_channel = 512
        out_size = (4, 4, 4)
    elif args.arch == 'c3d':
        base_model = C3D(num_classes=num_class, with_classifier=with_classifier)
        # args.logits_channel = 512
        out_size = (4, 4, 4)
    elif args.arch == 'r3d18':
        base_model = resnet18(num_classes=num_class, sample_size=int(args.ft_spatial_size), with_classifier=with_classifier)
        # args.logits_channel = 512
        out_size = (4, 4, 4)
    elif args.arch == 'r3d34':
        base_model = resnet34(num_classes=num_class, sample_size=int(args.ft_spatial_size), with_classifier=with_classifier)
        # args.logits_channel = 512
        out_size = (4, 4, 4)
    elif args.arch == 'r3d50':
        base_model = resnet50(num_classes=num_class, sample_size=int(args.ft_spatial_size), with_classifier=with_classifier)
        # args.logits_channel = 512
        out_size = (4, 4, 4)
    elif args.arch == 's3d':
        # base_model = S3D_G(num_class=num_class, drop_prob=args.dropout, in_channel=3)
        base_model = S3DG(num_classes=num_class, dropout_keep_prob=args.ft_dropout, input_channel=3, spatial_squeeze=True, with_classifier=True)
        # args.logits_channel = 1024
        out_size = (2, 7, 7)
    else:
        Exception("unsuporrted arch!")
    base_model = ft_load_weight(args, base_model)
    model = TCN(base_model,  out_size, args)
    model = nn.DataParallel(model).cuda()
    # cudnn.benchmark = True
    return model
