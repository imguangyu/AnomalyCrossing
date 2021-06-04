import torch.nn as nn
import torch
from torchvision.models.video.resnet import (
    BasicBlock,
    Bottleneck,
    BasicStem,
    R2Plus1dStem,
    _video_resnet,
)


__all__ = ["rgb_8f_xdc_video_encoder","rgb_16f_xdc_video_encoder", "rgb_32f_xdc_video_encoder"]

model_urls = {
    "r2plus1d_18_xdc_ig65m_kinetics": "https://github.com/HumamAlwassel/XDC/releases/download/model_weights/r2plus1d_18_xdc_ig65m_kinetics-f24f6ffb.pth",
    "r2plus1d_18_xdc_ig65m_random": "https://github.com/HumamAlwassel/XDC/releases/download/model_weights/r2plus1d_18_xdc_ig65m_random-189d23f4.pth",
    "r2plus1d_18_xdc_audioset": "https://github.com/HumamAlwassel/XDC/releases/download/model_weights/r2plus1d_18_xdc_audioset-f29ffe8f.pth",
    "r2plus1d_18_fs_kinetics": "https://github.com/HumamAlwassel/XDC/releases/download/model_weights/r2plus1d_18_fs_kinetics-622bdad9.pth",
    "r2plus1d_18_fs_imagenet": "https://github.com/HumamAlwassel/XDC/releases/download/model_weights/r2plus1d_18_fs_imagenet-ff446670.pth",
}

def xdc(pretraining='r2plus1d_18_xdc_ig65m_kinetics', progress=False, **kwargs):
    '''Pretrained video encoders as in 
    https://arxiv.org/abs/1911.12667

    Pretrained weights of all layers except the FC classifier layer are loaded. The FC layer 
    (of size 512 x num_classes) is randomly-initialized. Specify the keyword argument 
    `num_classes` based on your application (default is 400).

    Args:
        pretraining (string): The model pretraining type to load. Available pretrainings are
            r2plus1d_18_xdc_ig65m_kinetics: XDC pretrained on IG-Kinetics (default)
            r2plus1d_18_xdc_ig65m_random: XDC pretrained on IG-Random
            r2plus1d_18_xdc_audioset: XDC pretrained on AudioSet
            r2plus1d_18_fs_kinetics: fully-supervised Kinetics-pretrained baseline
            r2plus1d_18_fs_imagenet: fully-supervised ImageNet-pretrained baseline
        progress (bool): If True, displays a progress bar of the download to stderr
    '''
    assert pretraining in model_urls, \
        f'Unrecognized pretraining type. Available pretrainings: {list(model_urls.keys())}'
    
    model = r2plus1d_18(pretrained=False, progress=progress, **kwargs)

    state_dict = torch.hub.load_state_dict_from_url(
        model_urls[pretraining], progress=progress, check_hash=True,
    )

    model.load_state_dict(state_dict, strict=False)

    return model

class rgb_32f_xdc_video_encoder(nn.Module):
    def __init__(self, pretraining='r2plus1d_18_xdc_ig65m_kinetics', modelPath=''):
        super(rgb_32f_xdc_video_encoder, self).__init__()
        self.encoder=nn.Sequential(*list(
            xdc( pretraining=pretraining, progress=True).children())[:-2])
        
        for param in self.encoder.parameters():
            param.requires_grad = True

    def forward(self,x):
        x = self.encoder(x)
        return x

class rgb_16f_xdc_video_encoder(nn.Module):
    def __init__(self, pretraining='r2plus1d_18_xdc_ig65m_kinetics', modelPath=''):
        super(rgb_16f_xdc_video_encoder, self).__init__()
        self.encoder=nn.Sequential(*list(
            xdc( pretraining=pretraining, progress=True).children())[:-2])
        
        for param in self.encoder.parameters():
            param.requires_grad = True

    def forward(self,x):
        x = self.encoder(x)
        return x

class rgb_8f_xdc_video_encoder(nn.Module):
    def __init__(self, pretraining='r2plus1d_18_xdc_ig65m_kinetics', modelPath=''):
        super(rgb_8f_xdc_video_encoder, self).__init__()
        self.encoder=nn.Sequential(*list(
            xdc( pretraining=pretraining, progress=True).children())[:-2])
        
        for param in self.encoder.parameters():
            param.requires_grad = True

    def forward(self,x):
        x = self.encoder(x)
        return x

def r2plus1d_18(pretrained=False, progress=False, **kwargs):
    model = _video_resnet(
        "r2plus1d_18",
        False,
        False,
        block=BasicBlock,
        conv_makers=[Conv2Plus1D] * 4,
        layers=[2, 2, 2, 2],
        stem=R2Plus1dStem,
        **kwargs
    )
    # We need exact Caffe2 momentum for BatchNorm scaling
    for m in model.modules():
        if isinstance(m, nn.BatchNorm3d):
            m.eps = 1e-3
            m.momentum = 0.9
    return model


class BasicStem_Pool(nn.Sequential):
    def __init__(self):
        super(BasicStem_Pool, self).__init__(
            nn.Conv3d(
                3,
                64,
                kernel_size=(3, 7, 7),
                stride=(1, 2, 2),
                padding=(1, 3, 3),
                bias=False,
            ),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )


class Conv2Plus1D(nn.Sequential):
    def __init__(self, in_planes, out_planes, midplanes, stride=1, padding=1):

        midplanes = (in_planes * out_planes * 3 * 3 * 3) // (
            in_planes * 3 * 3 + 3 * out_planes
        )
        super(Conv2Plus1D, self).__init__(
            nn.Conv3d(
                in_planes,
                midplanes,
                kernel_size=(1, 3, 3),
                stride=(1, stride, stride),
                padding=(0, padding, padding),
                bias=False,
            ),
            nn.BatchNorm3d(midplanes),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                midplanes,
                out_planes,
                kernel_size=(3, 1, 1),
                stride=(stride, 1, 1),
                padding=(padding, 0, 0),
                bias=False,
            ),
        )

    @staticmethod
    def get_downsample_stride(stride):
        return (stride, stride, stride)
