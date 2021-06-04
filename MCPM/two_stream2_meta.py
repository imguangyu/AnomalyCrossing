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
# from models.utils import euclidean_metric
from models.classification_heads import protonet, ClassificationHead

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
parser.add_argument('--cls-head', default='protonet',
                    choices=['protonet','SVM-CS','SVM-He','SVM-WW','RidgeCS','Ridge','R2D2','Proto'],
                    help='classification head type')
parser.add_argument('--temperature', type=float, default=128,
                    help='protonet classification head annealing temperature')
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
parser.add_argument('--eps', default=0.0, type=float, metavar='N',
                    help='epsilon of label smoothing')
parser.add_argument('--label-smoothing', action='store_true',  
                   help='do label smoothing when training') 
parser.add_argument('--save-folder-id',  default='',
                    help='additional marker for the save folder') 
parser.add_argument('--iter-opt', action='store_true',  
                   help='do label smoothing when training') 
parser.add_argument('--lamda', default=1e-5, type=float, help="penalty factor for BSR")
parser.add_argument('--loss-type', default='CE',
                    choices=["CE", "BSR"],
                    help='training loss type')
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
    
    saveLocation="./checkpoint/"+args.dataset+"_"+args.arch+"_"+args.cls_head+"_split"+str(args.split)
    if args.save_folder_id:
        saveLocation += "_" + args.save_folder_id
    if not os.path.exists(saveLocation):
        os.makedirs(saveLocation)
    writer = SummaryWriter(saveLocation)
   
    # create model

    if args.evaluate:
        print("Building validation model ... ")
        encoder, cls_head = build_model_validate(args)
        #This line is not important, only dummy
        optimizer = AdamW(encoder.parameters(), lr= args.lr, weight_decay=args.weight_decay)
    elif training_continue:
        encoder, cls_head, start_epoch, optimizer, best_prec = build_model_continue(args)
        #lr = args.lr
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
            #param_group['lr'] = lr
        print("Continuing with best precision: %.3f and start epoch %d and lr: %f" %(best_prec,start_epoch,lr))
    else:
        print("Building model with SGD optimizer... ")
        encoder, cls_head = build_model(args)
        if args.cls_head == "protonet":
            optimizer = torch.optim.SGD(
                encoder.parameters(),
                lr=args.lr,
                momentum=args.momentum,
                dampening=0.9,
                weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.SGD(
               [{'params': encoder.parameters()}, 
                {'params': cls_head.parameters()}],
                lr=args.lr,
                momentum=args.momentum,
                dampening=0.9,
                weight_decay=args.weight_decay)
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
    #     optimizer, 'min', patience=2, verbose=True)
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
        train(train_loader_meta, encoder, cls_head, optimizer, epoch, modality, args)

        # evaluate on validation set
        accClassification = 0.0
        lossClassification = 0
        if (epoch + 1) % args.save_freq == 0:
            accClassification,lossClassification =  validate(val_loader_meta, encoder, cls_head, modality, args)
            writer.add_scalar('data/accuracy_validation', accClassification, epoch)
            writer.add_scalar('data/classification_loss_validation', lossClassification, epoch)
            # scheduler.step(lossClassification)
            scheduler.step()
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
                print("Save the best model")
                if hasattr(encoder,'module'):
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict_encoder': encoder.module.state_dict(),
                        'state_dict_cls_head': cls_head.state_dict(),
                        'best_prec': best_prec,
                        'best_loss': best_loss,
                        'optimizer' : optimizer.state_dict(),
                    }, is_best, checkpoint_name, saveLocation) 
                else:
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict_encoder': encoder.state_dict(),
                        'state_dict_cls_head': cls_head.state_dict(),
                        'best_prec': best_prec,
                        'best_loss': best_loss,
                        'optimizer' : optimizer.state_dict(),
                    }, is_best, checkpoint_name, saveLocation)
    
    checkpoint_name = "%04d_%s" % (epoch + 1, "checkpoint.pth.tar")
    if hasattr(encoder,'module'):
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict_encoder': encoder.module.state_dict(),
            'state_dict_cls_head': cls_head.state_dict(),
            'best_prec': best_prec,
            'best_loss': best_loss,
            'optimizer' : optimizer.state_dict(),
        }, is_best, checkpoint_name, saveLocation) 
    else:
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict_encoder': encoder.state_dict(),
            'state_dict_cls_head': cls_head.state_dict(),
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

    if args.cls_head == "protonet":
         cls_head = protonet(args.shot, args.way, args.temperature)
    else:
        cls_head = ClassificationHead(base_learner=args.cls_head, n_shot=args.shot, n_way=args.way)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            encoder=torch.nn.DataParallel(encoder)    
        encoder = encoder.cuda()
        cls_head = cls_head.cuda()
    return encoder, cls_head

def build_model_validate(args):
    modelLocation="./checkpoint/"+args.dataset+"_"+args.arch+"_"+args.cls_head+"_split"+str(args.split)
    if args.save_folder_id:
        modelLocation += "_" + args.save_folder_id
    model_path = os.path.join(modelLocation,'model_best.pth.tar') 
    params = torch.load(model_path)
    print(modelLocation)

    if args.dataset=='kinetics100':
        encoder=models.__dict__[args.arch](modelPath='')
    if args.cls_head == "protonet":
        cls_head = protonet(args.shot, args.way, args.temperature)
    else:
        cls_head = ClassificationHead(base_learner=args.cls_head, n_shot=args.shot, n_way=args.way)
        cls_head.load_state_dict(params['state_dict_cls_head'])

    encoder.load_state_dict(params['state_dict_encoder'])
    

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            encoder=torch.nn.DataParallel(encoder) 
        encoder.cuda()
        cls_head.cuda()

    encoder.eval() 
    cls_head.eval()
    return encoder, cls_head

def build_model_continue(args):
    modelLocation="./checkpoint/"+args.dataset+"_"+args.arch+"_"+args.cls_head+"_split"+str(args.split)
    if args.save_folder_id:
        modelLocation += "_" + args.save_folder_id
    model_path = os.path.join(modelLocation,'model_best.pth.tar') 
    params = torch.load(model_path)
    print(modelLocation)
    if args.dataset=='kinetics100':
        encoder=models.__dict__[args.arch](modelPath='')

    if args.cls_head == "protonet":
        cls_head = protonet(args.shot, args.way, args.temperature)
    else:
        cls_head = ClassificationHead(base_learner=args.cls_head, n_shot=args.shot, n_way=args.way)
        cls_head.load_state_dict(params['state_dict_cls_head'])
    
    encoder.load_state_dict(params['state_dict_encoder'])

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            encoder=torch.nn.DataParallel(encoder)    
        encoder = encoder.cuda()
        cls_head = cls_head.cuda()
    
    if args.cls_head == "protonet":
        optimizer = torch.optim.SGD(
            encoder.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            dampening=0.9,
            weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(
            [{'params': encoder.parameters()}, 
            {'params': cls_head.parameters()}],
            lr=args.lr,
            momentum=args.momentum,
            dampening=0.9,
            weight_decay=args.weight_decay)

    optimizer.load_state_dict(params['optimizer'])
    
    startEpoch = params['epoch']
    best_prec = params['best_prec']
    return encoder, cls_head, startEpoch, optimizer, best_prec

def train(train_loader, encoder, cls_head, optimizer, epoch, modality, args):
    batch_time = AverageMeter()
    lossesClassification = AverageMeter()
    acc = AverageMeter()
    
    # switch to train mode
    encoder.train()
    cls_head.train()

    end = time.time()

    #lable for the sampled tasks
    label = torch.arange(args.way).repeat(args.query)
    support_label = torch.arange(args.way).repeat(args.shot)
    if torch.cuda.is_available():
        label = label.type(torch.cuda.LongTensor)
        support_label = support_label.type(torch.cuda.LongTensor)
    else:
        label = label.type(torch.LongTensor)
        support_label = support_label.type(torch.LongTensor)

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
        
        #split the data to shot and query
        p = args.shot * args.way    
        data_shot, data_query = inputs[:p], inputs[p:]
        if torch.cuda.is_available():
            if HALF:
                data_shot = data_shot.cuda().half()
                # data_query = data_query.cuda().half()
            else:
                data_shot = data_shot.cuda()
                # data_query = data_query.cuda()
        
        if not args.iter_opt:
            if torch.cuda.is_available():
                if HALF:
                    data_query = data_query.cuda().half()
                else:
                    data_query = data_query.cuda()

            proto = encoder(data_shot)
            query = encoder(data_query)
            # move the suqeeze into the cls head models
            if args.loss_type == "BSR":
                logits, bsr = cls_head.forward_bsr(proto,query,support_label)
            else:
                logits = cls_head(proto,query,support_label)#.squeeze()

            if args.label_smoothing:
                smoothed_one_hot = one_hot(label.reshape(-1), args.way)
                smoothed_one_hot = smoothed_one_hot * (1 - args.eps) + (1 - smoothed_one_hot) * args.eps / (args.way - 1)
                log_prb = F.log_softmax(logits.reshape(-1, args.way), dim=1)
                loss = -(smoothed_one_hot * log_prb).sum(dim=1)
                loss = loss.mean()
            else:
                loss = F.cross_entropy(logits, label)

            # add bsr penalty
            if args.loss_type == "BSR":
                loss += args.lamda * bsr

            optimizer.zero_grad()
            loss.backward()
            # compute gradient and do SGD step
            optimizer.step()

            totalSamplePerIter =  logits.size(0)
            acc_mini_batch = count_acc(logits, label)
            acc.update(acc_mini_batch, totalSamplePerIter)
            loss_mini_batch_classification = loss.data.item()
            lossesClassification.update(loss_mini_batch_classification, totalSamplePerIter)
        else:
            optimizer.zero_grad()
            nq = data_query.size()[0]
            nb = torch.cuda.device_count()
            iter_size = max(nq // nb, 1)
            loss_mini_batch_classification = 0
            totalSamplePerIter = nq
            logits_all = []
            idx = 0
            while idx < nq:
                proto = encoder(data_shot)
                q = data_query[idx:idx+nb]
                q_label = label[idx:idx+nb]
                if torch.cuda.is_available():
                    if HALF:
                        q = q.cuda().half()
                    else:
                        q = q.cuda()
                query = encoder(q)
                # move the suqeeze into the cls head models
                if args.loss_type == "BSR":
                    logits, bsr = cls_head.forward_bsr(proto,query,support_label)
                else:
                    logits = cls_head(proto,query,support_label)
                if logits.dim() < 2:
                    logits = logits.unsqueeze(0)
                logits_all.append(logits)
                if args.label_smoothing:
                    smoothed_one_hot = one_hot(q_label.reshape(-1), args.way)
                    smoothed_one_hot = smoothed_one_hot * (1 - args.eps) + (1 - smoothed_one_hot) * args.eps / (args.way - 1)
                    log_prb = F.log_softmax(logits.reshape(-1, args.way), dim=1)
                    loss = -(smoothed_one_hot * log_prb).sum(dim=1)
                    loss = loss.mean() 
                else:
                    loss = F.cross_entropy(logits, q_label) 

                if args.loss_type == "BSR":
                    loss += args.lamda * bsr
                loss = loss / iter_size
                loss.backward()
                loss_mini_batch_classification += loss.data.item()
                idx += nb
            optimizer.step()

            logits_all = torch.cat(logits_all)
            acc_mini_batch = count_acc(logits_all, label)
            acc.update(acc_mini_batch, totalSamplePerIter)
            lossesClassification.update(loss_mini_batch_classification, totalSamplePerIter)
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i+1) % args.print_freq == 0:
            print('[%d] time: %.3f loss: %.4f acc: %.4f' %(i,batch_time.avg,lossesClassification.avg, acc.avg))

          
    print(' * Epoch: {epoch} Accuracy {acc.avg:.3f} Classification Loss {lossClassification.avg:.4f}\n'
          .format(epoch = epoch, acc=acc, lossClassification=lossesClassification))
          
    writer.add_scalar('data/classification_loss_training', lossesClassification.avg, epoch)
    writer.add_scalar('data/accuracy_training', acc.avg, epoch)

def validate(val_loader, encoder, cls_head, modality, args):
    batch_time = AverageMeter()
    lossesClassification = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    encoder.eval()
    cls_head.eval()

    #lable for the sampled tasks
    label = torch.arange(args.way).repeat(args.query)
    support_label = torch.arange(args.way).repeat(args.shot)
    if torch.cuda.is_available():
        label = label.type(torch.cuda.LongTensor)
        support_label = support_label.type(torch.cuda.LongTensor)
    else:
        label = label.type(torch.LongTensor)
        support_label = support_label.type(torch.LongTensor)
    
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
                
            #split the data to shot and query
            p = args.shot * args.way    
            data_shot, data_query = inputs[:p], inputs[p:]
            if torch.cuda.is_available():
                if HALF:
                    data_shot = data_shot.cuda().half()
                    # data_query = data_query.cuda().half()
                else:
                    data_shot = data_shot.cuda()
                    # data_query = data_query.cuda()
        
            if not args.iter_opt:
                # compute output
                if torch.cuda.is_available():
                    if HALF:
                        data_query = data_query.cuda().half()
                    else:
                        data_query = data_query.cuda()
                proto = encoder(data_shot)
                query = encoder(data_query)
                logits = cls_head(proto,query,support_label).squeeze()
                    
                if args.label_smoothing:
                    smoothed_one_hot = one_hot(label.reshape(-1), args.way)
                    smoothed_one_hot = smoothed_one_hot * (1 - args.eps) + (1 - smoothed_one_hot) * args.eps / (args.way - 1)
                    log_prb = F.log_softmax(logits.reshape(-1, args.way), dim=1)
                    loss = -(smoothed_one_hot * log_prb).sum(dim=1)
                    loss = loss.mean()
                else:
                    loss = F.cross_entropy(logits, label)
                # measure accuracy and record loss
                acc_mini_batch = count_acc(logits, label)
                
                lossesClassification.update(loss.data.item(), logits.size(0))
                acc.update(acc_mini_batch, logits.size(0))
            else:
                nq = data_query.size()[0]
                nb = torch.cuda.device_count()
                iter_size = max(nq // nb, 1)
                loss_mini_batch_classification = 0
                logits_all = []
                idx = 0
                while idx < nq:
                    proto = encoder(data_shot)
                    q = data_query[idx:idx+nb]
                    q_label = label[idx:idx+nb]
                    if torch.cuda.is_available():
                        if HALF:
                            q = q.cuda().half()
                        else:
                            q = q.cuda()
                    query = encoder(q)
                    logits = cls_head(proto,query,support_label)
                    if logits.dim() < 2:
                        logits = logits.unsqueeze(0)
                    logits_all.append(logits)
                    if args.label_smoothing:
                        smoothed_one_hot = one_hot(q_label.reshape(-1), args.way)
                        smoothed_one_hot = smoothed_one_hot * (1 - args.eps) + (1 - smoothed_one_hot) * args.eps / (args.way - 1)
                        log_prb = F.log_softmax(logits.reshape(-1, args.way), dim=1)
                        loss = -(smoothed_one_hot * log_prb).sum(dim=1)
                        loss = loss.mean() / iter_size
                    else:
                        loss = F.cross_entropy(logits, q_label) / iter_size
                    loss_mini_batch_classification += loss.data.item()
                    idx += nb
                logits_all = torch.cat(logits_all)
                acc_mini_batch = count_acc(logits_all, label)
                acc.update(acc_mini_batch, nq)
                lossesClassification.update(loss_mini_batch_classification, nq)
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

def one_hot(indices, depth):
    """
    Returns a one-hot tensor.
    This is a PyTorch equivalent of Tensorflow's tf.one_hot.
        
    Parameters:
      indices:  a (n_batch, m) Tensor or (m) Tensor.
      depth: a scalar. Represents the depth of the one hot dimension.
    Returns: a (n_batch, m, depth) Tensor or (m, depth) Tensor.
    """

    encoded_indicies = torch.zeros(indices.size() + torch.Size([depth])).cuda()
    index = indices.view(indices.size()+torch.Size([1]))
    encoded_indicies = encoded_indicies.scatter_(1,index,1)
    
    return encoded_indicies
    
if __name__ == '__main__':
    main()
