
import os
import time
import argparse
import shutil
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data

import video_transforms
import models
import datasets
from utils.model_path import rgb_3d_model_path_selection
import h5py


model_names = sorted(name for name in models.__dict__
    if not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Save Features Dota')
parser.add_argument('--settings', metavar='DIR', default='./datasets/settings/paperfold/TC102',
                    help='path to dataset setting files')
parser.add_argument('--dataset', '-d', default='paperfold',
                    choices=["paperfold"],
                    help='dataset: paperfold')
parser.add_argument('--frames-path', metavar='DIR', default='./datasets/dota/frames',
                    help='path to dataset files')    
parser.add_argument('--name-pattern-rgb',  default='frame%d.jpg',
                    help='name pattern of the frame files')                      
parser.add_argument('--arch', '-a', metavar='ARCH', default='rgb_r2plus1d_32f_34_encoder',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: rgb_resneXt3D64f101)')
parser.add_argument('-s', '--split', default=1, type=int, metavar='S',
                    help='which split of data to work on (default: 1)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=4, type=int,
                    metavar='N', help='mini-batch size (default: 50)')
parser.add_argument('--phase', default="train", help='phase of the split file, e.g. train, val or all')
parser.add_argument('--device', default="cuda", help='use cuda or cpu')
parser.add_argument('--modelLocation', default="", help='path of the saved model')
parser.add_argument('--saved-model-name', default="model_best", help='name of the saved model')
parser.add_argument('--outfile', help='output path of the extracted features')
parser.add_argument('--input-size', default=224, type=int,
                      help='input image size to the backbone')

class WrappedModel(nn.Module):
    def __init__(self, encoder):
        super(WrappedModel, self).__init__()
        self.encoder = encoder # that I actually define.
        self.avgpool = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
    def forward(self, x):
        out = self.avgpool(self.encoder(x)).squeeze()
        if len(out.size()) < 2:
            out.unsqueeze(0)
        return out


def load_encoder(args):
    
    encoder=models.__dict__[args.arch](modelPath='')
    if args.modelLocation:
        model_path = os.path.join(args.modelLocation,args.saved_model_name+'.pth.tar') 
        params = torch.load(model_path)
        print(args.modelLocation)
        encoder.load_state_dict(params['state_dict_encoder'])
    
    return encoder

def save_features(model, data_loader, modality, length, args,device):
    f = h5py.File(args.outfile, 'w')
    max_count = len(data_loader)*data_loader.batch_size
    all_labels = f.create_dataset('all_labels',(max_count,), dtype='i')
    # don'd required save image paths here
    # all_img_paths = f.create_dataset('all_img_paths',(max_count,), dtype='S200')
    all_feats=None
    count=0
    for i, (inputs,y) in enumerate(data_loader):
        if i%10 == 0:
            print('{:d}/{:d}'.format(i, len(data_loader)))
        if modality == "rgb" or modality == "pose":
            if "3D" in args.arch or 'r2plus1d' in args.arch or 'xdc' in args.arch or 'rep_flow' in args.arch or 'slowfast' in args.arch:
                inputs=inputs.view(-1,length,3,input_size,input_size).transpose(1,2)
            elif "tsm" in args.arch:
                inputs=inputs
            else:
                inputs=inputs.view(-1,3*length,input_size,input_size)
        elif modality == "flow":
            if "3D" in args.arch:
                inputs=inputs.view(-1,length,2,input_size,input_size).transpose(1,2)
            else:
                inputs=inputs.view(-1,2*length,input_size,input_size)            
        elif modality == "both":
            inputs=inputs.view(-1,5*length,input_size,input_size)
        
        inputs = inputs.to(device)
        # x_var = Variable(inputs)
        
        #dim of r2plus1d is (N, 512)
        feats = model(inputs)
        if all_feats is None:
            all_feats = f.create_dataset('all_feats', [max_count] + list( feats.size()[1:]) , dtype='f')
        all_feats[count:count+feats.size(0)] = feats.detach().cpu().numpy()#feats.data.cpu().numpy()
        all_labels[count:count+feats.size(0)] = y.cpu().numpy()
        # all_img_paths[count:count+feats.size(0)] =  [a.encode('utf8') for a in path]
        count = count + feats.size(0)

    count_var = f.create_dataset('count', (1,), dtype='i')
    count_var[0] = count

    f.close()

if __name__ == '__main__':
    args = parser.parse_args()
    input_size = int(args.input_size)
    width = input_size
    height = input_size

    cudnn.benchmark = True
    modality=args.arch.split('_')[0]

    if '64f' in args.arch:
        length=64
    elif '32f' in args.arch:
        length=32
    elif '8f' in args.arch:
        length=8
    elif '8f' in args.arch:
        length=8
    else:
        length=16

    print('length %d, img size %d' %(length, input_size))
    # Data transforming
    if modality == "rgb" or modality == "pose":
        is_color = True
        scale_ratios = [1.0, 0.875, 0.75, 0.66]
        if 'I3D' in args.arch:
            if 'resnet' in args.arch:
                clip_mean = [0.45, 0.45, 0.45] * length
                clip_std = [0.225, 0.225, 0.225] * length
            else:
                clip_mean = [0.5, 0.5, 0.5] * length
                clip_std = [0.5, 0.5, 0.5] * length
            #clip_std = [0.25, 0.25, 0.25] * args.num_seg * length
        elif 'MFNET3D' in args.arch:
            clip_mean = [0.48627451, 0.45882353, 0.40784314] * length
            clip_std = [0.234, 0.234, 0.234] * length
        elif "3D" in args.arch:
            clip_mean = [114.7748, 107.7354, 99.4750] * length
            clip_std = [1, 1, 1] * length
        elif "r2plus1d" in args.arch:
            clip_mean = [0.43216, 0.394666, 0.37645] * length
            clip_std = [0.22803, 0.22145, 0.216989] * length
        elif "xdc" in args.arch:
            clip_mean = [0.43216, 0.394666, 0.37645] * length
            clip_std = [0.22803, 0.22145, 0.216989] * length
        elif "rep_flow" in args.arch:
            clip_mean = [0.5, 0.5, 0.5] * length
            clip_std = [0.5, 0.5, 0.5] * length      
        elif "slowfast" in args.arch:
            clip_mean = [0.45, 0.45, 0.45] * length
            clip_std = [0.225, 0.225, 0.225] * length
        else:
            clip_mean = [0.485, 0.456, 0.406] * length
            clip_std = [0.229, 0.224, 0.225] * length
    elif modality == "pose":
        is_color = True
        scale_ratios = [1.0, 0.875, 0.75, 0.66]
        clip_mean = [0.485, 0.456, 0.406]
        clip_std = [0.229, 0.224, 0.225]
    elif modality == "flow":
        is_color = False
        scale_ratios = [1.0, 0.875, 0.75, 0.66]
        if 'I3D' in args.arch:
            clip_mean = [0.5, 0.5] * length
            clip_std = [0.5, 0.5] * length
        elif "3D" in args.arch:
            clip_mean = [127.5, 127.5] * length
            clip_std = [1, 1] * length        
        else:
            clip_mean = [0.5, 0.5] * length
            clip_std = [0.226, 0.226] * length
    elif modality == "both":
        is_color = True
        scale_ratios = [1.0, 0.875, 0.75, 0.66]
        clip_mean = [0.485, 0.456, 0.406, 0.5, 0.5] * length
        clip_std = [0.229, 0.224, 0.225, 0.226, 0.226] * length
    else:
        print("No such modality. Only rgb and flow supported.")

    crop = [228, 951, 265, 1060]
    normalize = video_transforms.Normalize(mean=clip_mean,
                                           std=clip_std)

    if "3D" in args.arch and not ('I3D' in args.arch or 'MFNET3D' in args.arch):  
        val_transform = video_transforms.Compose([
                video_transforms.ToTensor2(),
                normalize,
            ])
    else:
        val_transform = video_transforms.Compose([
                video_transforms.ToTensor(),
                normalize,
            ])

    setting_file = "%s_%s_%df_split%d.txt" % (args.phase, modality, length, args.split)
    split_file = os.path.join(args.settings, setting_file)
    print(split_file)
    if not os.path.exists(split_file):
        print("No split file exists in %s directory. Preprocess the dataset first" % (os.path.join(args.settings,args.dataset)))
    
    eval_dataset = datasets.__dict__[args.dataset](root=args.frames_path,
                                                source=split_file,
                                                modality=modality,
                                                name_pattern=args.name_pattern_rgb,
                                                is_color=is_color,
                                                new_width=width,
                                                new_height=height,
                                                crop=crop,
                                                video_transform=val_transform
                                                )
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    device = torch.device(args.device)
    encoder = load_encoder(args)
    if args.device == "cuda" and torch.cuda.device_count() > 1:
        encoder=torch.nn.DataParallel(encoder) 
    model = WrappedModel(encoder)
    model.to(device)
    model.eval()

    dirname = os.path.dirname(args.outfile)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    save_features(model, eval_loader, modality, length, args, device)