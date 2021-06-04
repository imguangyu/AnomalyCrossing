
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

# number of clips for diff cls with ego
# {'lateral': 643,
#     'leave_to_left': 266,
#     'leave_to_right': 203,
#     'moving_ahead_or_waiting': 592,
#     'obstacle': 64,
#     'oncoming': 528,
#     'pedestrian': 52,
#     'start_stop_or_stationary': 66,
#     'turning': 1330,
#     'unknown': 56})

# number of clips for diff cls with unego
# {'lateral': 330,
# 'leave_to_left': 269,
# 'leave_to_right': 280,
# 'moving_ahead_or_waiting': 339,
# 'obstacle': 54,
# 'oncoming': 154,
# 'pedestrian': 70,
# 'start_stop_or_stationary': 68,
# 'turning': 1066,
# 'unknown': 56}

DOTA_CLASSES = ['lateral','leave_to_left','leave_to_right',
               'moving_ahead_or_waiting','obstacle','oncoming',
               'pedestrian','start_stop_or_stationary','turning',
               'unknown']

model_names = sorted(name for name in models.__dict__
    if not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Save Features Dota')
parser.add_argument('--settings', metavar='DIR', default='./datasets/settings',
                    help='path to dataset setting files')
parser.add_argument('--dataset', '-d', default='dota',
                    choices=["dota", "ucf_crime"],
                    help='dataset: dota')
parser.add_argument('--frames-path', metavar='DIR', default='./datasets/dota/frames',
                    help='path to dataset files')    
parser.add_argument('--name-pattern-rgb',  default='%06d.jpg',
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
parser.add_argument('--num-seg', default=1, type=int,
                    metavar='N', help='Number of segments for temporal LSTM (default: 16)')
parser.add_argument('-b', '--batch-size', default=4, type=int,
                    metavar='N', help='mini-batch size (default: 50)')
parser.add_argument('--phase', default="all", help='phase of the split file, e.g. train, val or all')
parser.add_argument('--device', default="cuda", help='use cuda or cpu')
parser.add_argument('--modelLocation', default="", help='path of the saved model')
parser.add_argument('--saved-model-name', default="model_best", help='name of the saved model')
parser.add_argument('--outfile', help='output path of the extracted features')
parser.add_argument('--selected-cls', default="", help="Select one special class")
                    # choices=DOTA_CLASSES,
                    # help="Select one class from: " +
                    #     " | ".join(DOTA_CLASSES) +
                    #     " (default: '')")
parser.add_argument('--ego-envolve', default="",
                    choices=["True","False"],
                    help='Specify use only ego or non-ego or both (default both)')
parser.add_argument('--scale', default=None, type=float,
                      help='customized image size scale')
parser.add_argument('--width', default=340, type=int,
                      help='resize image to this width')
parser.add_argument('--height', default=256, type=int,
                      help='resize image to this height')
parser.add_argument('--clip-length', default=-1, type=int,
                      help='clip length enforced by the user')

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
        #please provide the saved model name with the file extension as well
        #to make the code be mode general
        model_path = os.path.join(args.modelLocation,args.saved_model_name)#+'.pth.tar' 
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
    if '3D' in args.arch:
        if 'I3D' in args.arch or 'MFNET3D' in args.arch:
            if '112' in args.arch:
                scale = 0.5
            else:
                scale = 1
        else:
            if '224' in args.arch:
                scale = 1
            else:
                scale = 0.5
    elif 'r2plus1d' in args.arch:
        scale = 0.5
    else:
        scale = 1
    
    if args.scale:
        scale = args.scale

    input_size = int(224 * scale)
    width = int(args.width * scale)#int(340 * scale)
    height = int(args.height * scale)#int(256 * scale)

    cudnn.benchmark = True
    modality = args.arch.split('_')[0]

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
    if args.clip_length > 0:
        length = args.clip_length

    print('length %d, input size %d, width %d height %d' %(length, input_size, width, height))
    # Data transforming
    if modality == "rgb" or modality == "pose":
        is_color = True
        scale_ratios = [1.0, 0.875, 0.75, 0.66]
        if 'I3D' in args.arch:
            if 'resnet' in args.arch:
                clip_mean = [0.45, 0.45, 0.45] * args.num_seg * length
                clip_std = [0.225, 0.225, 0.225] * args.num_seg * length
            else:
                clip_mean = [0.5, 0.5, 0.5] * args.num_seg * length
                clip_std = [0.5, 0.5, 0.5] * args.num_seg * length
            #clip_std = [0.25, 0.25, 0.25] * args.num_seg * length
        elif 'MFNET3D' in args.arch:
            clip_mean = [0.48627451, 0.45882353, 0.40784314] * args.num_seg * length
            clip_std = [0.234, 0.234, 0.234]  * args.num_seg * length
        elif "3D" in args.arch:
            clip_mean = [114.7748, 107.7354, 99.4750] * args.num_seg * length
            clip_std = [1, 1, 1] * args.num_seg * length
        elif "r2plus1d" in args.arch:
            clip_mean = [0.43216, 0.394666, 0.37645] * args.num_seg * length
            clip_std = [0.22803, 0.22145, 0.216989] * args.num_seg * length
        elif "xdc" in args.arch:
            clip_mean = [0.43216, 0.394666, 0.37645] * args.num_seg * length
            clip_std = [0.22803, 0.22145, 0.216989] * args.num_seg * length
        elif "rep_flow" in args.arch:
            clip_mean = [0.5, 0.5, 0.5] * args.num_seg * length
            clip_std = [0.5, 0.5, 0.5] * args.num_seg * length      
        elif "slowfast" in args.arch:
            clip_mean = [0.45, 0.45, 0.45] * args.num_seg * length
            clip_std = [0.225, 0.225, 0.225] * args.num_seg * length
        else:
            clip_mean = [0.485, 0.456, 0.406] * args.num_seg * length
            clip_std = [0.229, 0.224, 0.225] * args.num_seg * length
    elif modality == "pose":
        is_color = True
        scale_ratios = [1.0, 0.875, 0.75, 0.66]
        clip_mean = [0.485, 0.456, 0.406] * args.num_seg
        clip_std = [0.229, 0.224, 0.225] * args.num_seg
    elif modality == "flow":
        is_color = False
        scale_ratios = [1.0, 0.875, 0.75, 0.66]
        if 'I3D' in args.arch:
            clip_mean = [0.5, 0.5] * args.num_seg * length
            clip_std = [0.5, 0.5] * args.num_seg * length
        elif "3D" in args.arch:
            clip_mean = [127.5, 127.5] * args.num_seg * length
            clip_std = [1, 1] * args.num_seg * length        
        else:
            clip_mean = [0.5, 0.5] * args.num_seg * length
            clip_std = [0.226, 0.226] * args.num_seg * length
    elif modality == "both":
        is_color = True
        scale_ratios = [1.0, 0.875, 0.75, 0.66]
        clip_mean = [0.485, 0.456, 0.406, 0.5, 0.5] * args.num_seg * length
        clip_std = [0.229, 0.224, 0.225, 0.226, 0.226] * args.num_seg * length
    else:
        print("No such modality. Only rgb and flow supported.")

    
    normalize = video_transforms.Normalize(mean=clip_mean,
                                           std=clip_std)

    if "3D" in args.arch and not ('I3D' in args.arch or 'MFNET3D' in args.arch):  
        val_transform = video_transforms.Compose([
                video_transforms.CenterCrop((input_size)),
                video_transforms.ToTensor2(),
                normalize,
            ])
    else:
        val_transform = video_transforms.Compose([
                video_transforms.CenterCrop((input_size)),
                video_transforms.ToTensor(),
                normalize,
            ])
    setting_file = "%s_%s_split%d.txt" % (args.phase, modality, args.split)
    split_file = os.path.join(args.settings, args.dataset, setting_file)
    if not os.path.exists(split_file):
        print("No split file exists in %s directory. Preprocess the dataset first" % (os.path.join(args.settings,args.dataset)))
    
    if args.selected_cls:
        selected_cls = args.selected_cls
    else:
        selected_cls = None 
    
    if args.ego_envolve:
        ego_envolve = args.ego_envolve
    else:
        ego_envolve = None 
    
    print("selected_cls: {}; ego_envolve: {}".format(selected_cls, ego_envolve))

    eval_dataset = datasets.__dict__[args.dataset](root=args.frames_path,
                                                source=split_file,
                                                phase="val",
                                                modality=modality,
                                                ego_envolve=ego_envolve,
                                                selected_cls=selected_cls,
                                                name_pattern=args.name_pattern_rgb,
                                                is_color=is_color,
                                                new_length=length,
                                                new_width=width,
                                                new_height=height,
                                                video_transform=val_transform,
                                                num_segments=args.num_seg)

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