#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 13:15:15 2019

@author: esat
"""


import os
import time
import argparse
import shutil
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from tensorboardX import SummaryWriter
import torch.nn.functional as F

from torch.optim import lr_scheduler

import video_transforms
import models
import datasets
from datasets.samplers import CategoriesSampler
import swats

from opt.AdamW import AdamW
from utils.model_path import rgb_3d_model_path_selection
from models.gnn import GNN_nl
from models.gnnnet import GnnNet

model_names = sorted(name for name in models.__dict__
    if not name.startswith("__")
    and callable(models.__dict__[name]))

dataset_names = sorted(name for name in datasets.__all__)

parser = argparse.ArgumentParser(description='PyTorch Two-Stream2 Meta')
parser.add_argument('--settings', metavar='DIR', default='./datasets/settings',
                    help='path to dataset setting files')
parser.add_argument('--dataset', '-d', default='kinetics100',
                    choices=["kinetics100"],
                    help='dataset: kinetics100')
parser.add_argument('--frames-path', metavar='DIR', default='./datasets/kinetcis100_frames',
                    help='path to dataset files')    
parser.add_argument('--name-pattern-rgb',  default='image_%05d.jpg',
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
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', '--learning-rate', default=1e-2, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-3, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', default=1, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--save-freq', default=1, type=int,
                    metavar='N', help='save frequency (default: 25)')
parser.add_argument('--num-seg', default=1, type=int,
                    metavar='N', help='Number of segments for temporal LSTM (default: 16)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('-c', '--continue', dest='contine', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--num-tasks-train', default=100, type=int, metavar='N',
                    help='number of training tasks per epsisode (default 100)')
parser.add_argument('--num-tasks-val', default=500, type=int, metavar='N',
                    help='number of validation tasks (default 500)')
parser.add_argument('--way', default=5, type=int, metavar='N',
                    help='number of ways in the few-shot learning setting')
parser.add_argument('--shot', default=5, type=int, metavar='N',
                    help='number of shots in the few-shot learning setting')
parser.add_argument('--query', default=5, type=int, metavar='N',
                    help='number of queries in the few-shot learning setting')
parser.add_argument('--feat-dim', default=512, type=int, metavar='N',
                    help='feature dimension from the backbone model, e.g. 512 for r2plus1d')
parser.add_argument('--gnn-nf', default=96, type=int, metavar='N',
                    help='feature dimension of the GNN')
parser.add_argument('--gnn-node-dim', default=128, type=int, metavar='N',
                    help='dimension of the GNN node')
parser.add_argument('--fine-tune', action='store_true',  help='fine tuning during training ') 
parser.add_argument('--save-folder-id',  default='',
                    help='additional marker for the save folder') 

best_prec = -1
best_loss = 300000

HALF = False

select_according_to_best_classsification_lost = False #Otherwise select according to top1 default: False

training_continue = False

def main():
    global best_prec, writer, best_loss, length, width, height, input_size
 
    args = parser.parse_args()
    training_continue = args.contine
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
        
    print('scale: %.1f' %(scale))
    
    input_size = int(224 * scale)
    width = int(340 * scale)
    height = int(256 * scale)
    
    saveLocation="./checkpoint/"+args.dataset+"_"+args.arch+"_gnnnet"+"_split"+str(args.split)
    if args.save_folder_id:
        saveLocation += "_" + args.save_folder_id

    if not os.path.exists(saveLocation):
        os.makedirs(saveLocation)
    writer = SummaryWriter(saveLocation)
   
    # create model

    if args.evaluate:
        print("Building validation model ... ")
        model = build_model_validate(args)
        #This line is not important, only dummy
        optimizer = AdamW(model.parameters(), lr= args.lr, weight_decay=args.weight_decay)
    elif training_continue:
        model, start_epoch, optimizer, best_prec = build_model_continue(args)
        #lr = args.lr
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
            #param_group['lr'] = lr
        print("Continuing with best precision: %.3f and start epoch %d and lr: %f" %(best_prec,start_epoch,lr))
    else:
        print("Building model with SGD optimizer... ")
        model= build_model(args)
        # optimizer = torch.optim.SGD(
        #         model.parameters(),
        #         lr=args.lr,
        #         momentum=args.momentum,
        #         dampening=0.9,
        #         weight_decay=args.weight_decay)
        optimizer = AdamW(model.parameters(), lr= args.lr, weight_decay=args.weight_decay)

        start_epoch = 0
    
    if torch.cuda.is_available():
        if HALF:
            encoder.half()  # convert to half precision
            for layer in encoder.modules():
                if isinstance(layer, nn.BatchNorm2d):
                    layer.float()
            cls_head.half()  # convert to half precision
            for layer in cls_head.modules():
                if isinstance(layer, nn.BatchNorm2d):
                    layer.float()
    
    print("Model %s is loaded. " % (args.arch))

    # define loss function (criterion) and optimizer
    # criterion = nn.CrossEntropyLoss().cuda()
    

    # optimizer = AdamW(model.parameters(),
    #                   lr=args.lr,
    #                   weight_decay=args.weight_decay)
    
    # scheduler = lr_scheduler.ReduceLROnPlateau(
    #     optimizer, 'min', patience=1, verbose=True)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    #optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    #optimizer = swats.SWATS(model.parameters(), args.lr)
    
    
    print("Saving everything to directory %s." % (saveLocation))
    if args.dataset=='kinetics100':
        print("Meta Train on Kinetics 100!")
    else:
        print("No convenient dataset entered, exiting....")
        return 0
    
    cudnn.benchmark = True
    modality=args.arch.split('_')[0]

    if '64f' in args.arch:
        length=64
    elif '32f' in args.arch:
        length=32
    elif '8f' in args.arch:
        length=8
    else:
        length=16

    print('length %d' %(length))
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
        train_transform = video_transforms.Compose([
                video_transforms.MultiScaleCrop((input_size, input_size), scale_ratios),
                video_transforms.RandomHorizontalFlip(),
                video_transforms.ToTensor2(),
                normalize,
            ])
    
        val_transform = video_transforms.Compose([
                video_transforms.CenterCrop((input_size)),
                video_transforms.ToTensor2(),
                normalize,
            ])
    else:
        train_transform = video_transforms.Compose([
                video_transforms.MultiScaleCrop((input_size, input_size), scale_ratios),
                video_transforms.RandomHorizontalFlip(),
                video_transforms.ToTensor(),
                normalize,
            ])
    
        val_transform = video_transforms.Compose([
                video_transforms.CenterCrop((input_size)),
                video_transforms.ToTensor(),
                normalize,
            ])

    # data loading
    train_setting_file = "train_%s_split%d.txt" % (modality, args.split)
    train_split_file = os.path.join(args.settings, args.dataset, train_setting_file)
    val_setting_file = "val_%s_split%d.txt" % (modality, args.split)
    val_split_file = os.path.join(args.settings, args.dataset, val_setting_file)
    if not os.path.exists(train_split_file) or not os.path.exists(val_split_file):
        print("No split file exists in %s directory. Preprocess the dataset first" % (os.path.join(args.settings,args.dataset)))

    train_dataset = datasets.__dict__[args.dataset](root=args.frames_path,
                                                source=train_split_file,
                                                phase="train",
                                                modality=modality,
                                                name_pattern=args.name_pattern_rgb,
                                                is_color=is_color,
                                                new_length=length,
                                                new_width=width,
                                                new_height=height,
                                                video_transform=train_transform,
                                                num_segments=args.num_seg)
    
    val_dataset = datasets.__dict__[args.dataset](root=args.frames_path,
                                                source=val_split_file,
                                                phase="val",
                                                modality=modality,
                                                name_pattern=args.name_pattern_rgb,
                                                is_color=is_color,
                                                new_length=length,
                                                new_width=width,
                                                new_height=height,
                                                video_transform=val_transform,
                                                num_segments=args.num_seg)

    print('{} samples found, {} train samples and {} test samples.'.format(len(val_dataset)+len(train_dataset),
                                                                           len(train_dataset),
                                                                           len(val_dataset)))

    

    train_sampler = CategoriesSampler(train_dataset.labels, args.num_tasks_train, args.way, args.shot + args.query)
    val_sampler = CategoriesSampler(val_dataset.labels, args.num_tasks_val, args.way, args.shot + args.query)


    train_loader_meta = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=args.workers, pin_memory=True)

    val_loader_meta = torch.utils.data.DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        accurcy,lossClassification = validate(val_loader_meta, encoder, cls_head, modality, args)
        return

    for epoch in range(start_epoch, args.epochs):

        # train for one epoch
        accClassification, lossClassification = train(train_loader_meta, model, optimizer, epoch, modality, args)
        scheduler.step()
        # scheduler.step(lossClassification)
        # evaluate on validation set
        # accClassification = 0.0
        # lossClassification = 0
        # if (epoch + 1) % args.save_freq == 0:
            # accClassification,lossClassification =  validate(val_loader_meta, model, modality, args)
            # writer.add_scalar('data/accuracy_validation', accClassification, epoch)
            # writer.add_scalar('data/classification_loss_validation', lossClassification, epoch)
            # scheduler.step(lossClassification)
        # remember best prec@1 and save checkpoint
        
        if select_according_to_best_classsification_lost:
            is_best = lossClassification < best_loss
        else:
            is_best = accClassification > best_prec

        best_loss = min(lossClassification, best_loss)
        best_prec = max(accClassification, best_prec)
        


        if (epoch + 1) % args.save_freq == 0:
            checkpoint_name = "%04d_%s" % (epoch + 1, "checkpoint.pth.tar")
            if is_best:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict_encoder': model.encoder_without_dp.state_dict(),
                    'state_dict_fc': model.fc.state_dict(),
                    'state_dict_gnn': model.gnn.state_dict(),
                    'best_prec': best_prec,
                    'best_loss': best_loss,
                    'optimizer' : optimizer.state_dict(),
                }, is_best, checkpoint_name, saveLocation)
    
    checkpoint_name = "%04d_%s" % (epoch + 1, "checkpoint.pth.tar")
    save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict_encoder': model.encoder_without_dp.state_dict(),
                    'state_dict_fc': model.fc.state_dict(),
                    'state_dict_gnn': model.gnn.state_dict(),
                    'best_prec': best_prec,
                    'best_loss': best_loss,
                    'optimizer' : optimizer.state_dict(),
                }, is_best, checkpoint_name, saveLocation)
    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()

def build_model(args):
    modality=args.arch.split('_')[0]
    if modality == "rgb":
        model_path = rgb_3d_model_path_selection(args.arch)
    elif modality == "pose":
        model_path = rgb_3d_model_path_selection(args.arch)       
    elif modality == "flow":
        model_path=''
        if "3D" in args.arch:
            if 'I3D' in args.arch:
                 model_path='./weights/flow_imagenet.pth'   
            elif '3D' in args.arch:
                 model_path='./weights/Flow_Kinetics_64f.pth'   
    elif modality == "both":
        model_path='' 
        
    if args.dataset=='kinetics100':
        print('model path is: %s' %(model_path))
        encoder = models.__dict__[args.arch](modelPath=model_path)
    model = GnnNet(encoder, args.feat_dim, args.way, args.shot, args.query, args.gnn_node_dim, args.gnn_nf)
     
    if torch.cuda.is_available():
        model = model.cuda()
    return model

# Fix these two func later
def build_model_validate(args):
    modelLocation="./checkpoint/"+args.dataset+"_"+args.arch+"_gnnnet"+"_split"+str(args.split)
    if args.save_folder_id:
        modelLocation += "_" + args.save_folder_id
    model_path = os.path.join(modelLocation,'model_best.pth.tar') 
    params = torch.load(model_path)
    print(modelLocation)

    if args.dataset=='kinetics100':
        encoder=models.__dict__[args.arch](modelPath='')

    model = GnnNet(encoder, args.feat_dim, args.way, args.shot, args.query, args.gnn_node_dim, args.gnn_nf)
    model.encoder_without_dp.load_state_dict(params['state_dict_encoder'])
    model.fc.load_state_dict(params['state_dict_fc'])
    model.gnn.load_state_dict(params['state_dict_gnn'])

    if torch.cuda.is_available():
        model.cuda()

    return model

def build_model_continue(args):
    modelLocation="./checkpoint/"+args.dataset+"_"+args.arch+"_gnnnet"+"_split"+str(args.split)
    if args.save_folder_id:
        modelLocation += "_" + args.save_folder_id
    model_path = os.path.join(modelLocation,'model_best.pth.tar') 
    params = torch.load(model_path)
    print(modelLocation)
    if args.dataset=='kinetics100':
        encoder=models.__dict__[args.arch](modelPath='')
    model = GnnNet(encoder, args.feat_dim, args.way, args.shot, args.query, args.gnn_node_dim, args.gnn_nf)
    
    model.encoder_without_dp.load_state_dict(params['state_dict_encoder'])
    model.fc.load_state_dict(params['state_dict_fc'])
    model.gnn.load_state_dict(params['state_dict_gnn'])
    
    if torch.cuda.is_available():
        model.cuda()
    
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        dampening=0.9,
        weight_decay=args.weight_decay)
    optimizer.load_state_dict(params['optimizer'])
    
    startEpoch = params['epoch']
    best_prec = params['best_prec']
    return model, startEpoch, optimizer, best_prec

def train(train_loader, model, optimizer, epoch, modality, args):
    batch_time = AverageMeter()
    lossesClassification = AverageMeter()
    acc = AverageMeter()
    
    # switch to train mode
    model.train()

    end = time.time()

    #lable for the sampled tasks
    label = torch.from_numpy(np.repeat(range(args.way), args.query))
    if torch.cuda.is_available():
        label = label.cuda()

    for i, (inputs, _) in enumerate(train_loader):
        if modality == "rgb" or modality == "pose":
            if "3D" in args.arch or 'r2plus1d' in args.arch or 'rep_flow' in args.arch or 'slowfast' in args.arch:
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
        
        if torch.cuda.is_available():
            if HALF:
                inputs = inputs.cuda().half()
            else:
                inputs = inputs.cuda()
        
        optimizer.zero_grad()
        if args.fine_tune:
            logits = model.set_forward_finetune_without_mamal(inputs)
        else:
            logits = model.set_forward(inputs)

        loss = F.cross_entropy(logits, label)
        loss.backward()
        # compute gradient and do SGD step
        optimizer.step()

        totalSamplePerIter =  logits.size(0)
        acc_mini_batch = count_acc(logits, label)
        acc.update(acc_mini_batch, totalSamplePerIter)
        loss_mini_batch_classification = loss.data.item()
        lossesClassification.update(loss_mini_batch_classification, totalSamplePerIter)

        batch_time.update(time.time() - end)
        end = time.time()

        if (i+1) % args.print_freq == 0:
            print('[%d] time: %.3f loss: %.4f acc: %.4f' %(i,batch_time.avg,lossesClassification.avg, acc.avg))

          
    print(' * Epoch: {epoch} Accuracy {acc.avg:.3f} Classification Loss {lossClassification.avg:.4f}\n'
          .format(epoch = epoch, acc=acc, lossClassification=lossesClassification))
          
    writer.add_scalar('data/classification_loss_training', lossesClassification.avg, epoch)
    writer.add_scalar('data/accuracy_training', acc.avg, epoch)
    return acc.avg, lossesClassification.avg

def validate(val_loader, model, modality, args):
    batch_time = AverageMeter()
    lossesClassification = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()


    #lable for the sampled tasks
    label = torch.from_numpy(np.repeat(range(args.way), args.query))
    if torch.cuda.is_available():
        label = label.cuda()

    end = time.time()
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(val_loader):
            if modality == "rgb" or modality == "pose":
                if "3D" in args.arch or 'r2plus1d' in args.arch or 'rep_flow' in args.arch or 'slowfast' in args.arch:
                    inputs=inputs.view(-1,length,3,input_size,input_size).transpose(1,2)
                elif "tsm" in args.arch:
                    inputs = inputs
                else:
                    inputs=inputs.view(-1,3*length,input_size,input_size)
            elif modality == "flow":
                if "3D" in args.arch:
                    inputs=inputs.view(-1,length,2,input_size,input_size).transpose(1,2)
                else:
                    inputs=inputs.view(-1,2*length,input_size,input_size)      
            elif modality == "both":
                inputs=inputs.view(-1,5*length,input_size,input_size)
                
            if torch.cuda.is_available():
                if HALF:
                    inputs = inputs.cuda().half()
                else:
                    inputs = inputs.cuda()
            
            #note for eveluation we don't update the model
            #so only no fine tuning will be used
            logits = model.set_forward(inputs)
            loss = F.cross_entropy(logits, label)
            # measure accuracy and record loss
            acc_mini_batch = count_acc(logits, label)
            
            lossesClassification.update(loss.data.item(), logits.size(0))
            acc.update(acc_mini_batch, logits.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
        print(' * * Validation Accuracy {acc.avg:.3f} Classification Loss {lossClassification.avg:.4f}\n' 
              .format(acc=acc, lossClassification=lossesClassification))

    return acc.avg,  lossesClassification.avg

def save_checkpoint(state, is_best, filename, resume_path):
    cur_path = os.path.join(resume_path, filename)
    torch.save(state, cur_path)
    best_path = os.path.join(resume_path, 'model_best.pth.tar')
    if is_best:
        shutil.copyfile(cur_path, best_path)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    else:
        return (pred == label).type(torch.FloatTensor).mean().item()

if __name__ == '__main__':
    main()
