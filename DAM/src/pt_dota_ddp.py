from __future__ import print_function

import sys
import time
import torch
import torch.backends.cudnn as cudnn
from utils.utils import AverageMeter
from data.config import pt_data_config, pt_augmentation_config
from data.dataloader import pt_data_loader_init, pt_data_loader_init_ddp
import torch.nn as nn
from augment.gen_positive import GenPositive
from augment.gen_negative import GenNegative
from utils.learning_rate_adjust import pt_adjust_learning_rate
from utils.moment_update import moment_update
from model.config import pt_model_config_ddp
from loss.config import pt_optim_init
from utils.load_weights import pt_load_weight
from utils.utils import Timer
from datetime import datetime
from loss.tcr import tcr
import utils.utils as utils


def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]


def shift(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(1, x.size(dim)+1, 1,
                                dtype=torch.long, device=x.device)
    indices[dim] %= x.size(dim)
    # print(indices[dim])
    return x[tuple(indices)]


def get_shuffle_ids(bsz):
    """generate shuffle ids for ShuffleBN"""
    forward_inds = torch.randperm(bsz).long().cuda()
    backward_inds = torch.zeros(bsz).long().cuda()
    value = torch.arange(bsz).long().cuda()
    backward_inds.index_copy_(0, forward_inds, value)
    return forward_inds, backward_inds


def pretext_train_dota(args, recorder):
    #initialize the DDP Settings
    assert args.pt_method in ['dsm', 'moco'], "DDP only support dsm and moco"
    
    device = torch.device(args.device)

    num_class, data_length, image_tmpl = pt_data_config(args)
    # print("tp_length is: ", data_length)
    train_transforms, test_transforms, eval_transforms = pt_augmentation_config(args)
    if args.distributed:
        train_loader, val_loader, eval_loader, train_samples, val_samples, eval_samples, train_sampler, val_sampler, eval_sampler = pt_data_loader_init_ddp(args,
                                                                                                                                    data_length,
                                                                                                                                    image_tmpl,
                                                                                                                                    train_transforms,
                                                                                                                                    test_transforms,
                                                                                                                                    eval_transforms) 
    else:
        train_loader, val_loader, eval_loader, train_samples, val_samples, eval_samples = pt_data_loader_init(args,
                                                                                                       data_length,
                                                                                                       image_tmpl,
                                                                                                       train_transforms,
                                                                                                       test_transforms,
                                                                                                       eval_transforms)

    n_data = len(train_loader)
    
    #build models
    model, model_ema = pt_model_config_ddp(args, num_class)
    model = model.to(device)
    model_ema = model_ema.to(device)

    # == optim config==
    contrast, criterion, optimizer = pt_optim_init(args, model, n_data)
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model_ema = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_ema)
    
    model_without_ddp = model
    if args.distributed:
        print("use gpu: {}".format(args.gpu_id))
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu_id], output_device=args.gpu_id)
        model_without_ddp = model.module
        model_ema = torch.nn.parallel.DistributedDataParallel(model_ema, device_ids=[args.gpu_id], output_device=args.gpu_id)
    # == load weights ==
    # model, model_ema = pt_load_weight(args, model, model_ema, optimizer, contrast)
    # ZL: I believe this will work for the distributed models but need to double check, how?
    moment_update(model, model_ema, 0)
    # if args.pt_method in ['dsm', 'moco']:
    #     model_ema = model_ema.cuda()
    # # copy weights from `model' to `model_ema'
    #     moment_update(model, model_ema, 0)
    cudnn.benchmark = True
    # optionally resume from a checkpoint
    args.start_epoch = 1

    # ==================================== our data augmentation method=================================
    if args.pt_method in ['dsm', 'dsm_triplet']:
        pos_aug = GenPositive()
        neg_aug = GenNegative()

    # =======================================add message =====================
    if utils.is_main_process():
        recorder.record_message('a', '='*100)
        recorder.record_message('a', '-'*40+'pretrain'+'-'*40)
        recorder.record_message('a', '='*100)
    # ====================update lr_decay from str to numpy=========
    iterations = args.pt_lr_decay_epochs.split(',')
    args.pt_lr_decay_epochs = list([])
    for it in iterations:
        args.pt_lr_decay_epochs.append(int(it))
    timer = Timer()
    # routine
    print('*'*70+'Step1: pretrain'+'*'*20 + '*'*50)
    for epoch in range(args.pt_start_epoch, args.pt_epochs + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        timer.tic()
        pt_adjust_learning_rate(epoch, args, optimizer)
        print("==> training...")

        time1 = time.time()
        if args.pt_method == "moco":
            loss, prob = train_moco(device, epoch, train_loader, model, model_ema, contrast, criterion, optimizer, args, recorder)
        elif args.pt_method == "dsm":
             loss, prob = train_dsm(device, epoch, train_loader, model, model_ema, contrast, criterion, optimizer, args, pos_aug,
                                     neg_aug, recorder)
            # loss, prob = epoch * 0.01, 0.02*epoch
        elif args.pt_method == "dsm_triplet":
            loss = train_dsm_triplet(device, epoch, train_loader, model, optimizer, args, pos_aug, neg_aug, recorder)
        else:
            Exception("Not support method now!")
        # loss=0 # For test only 
        if utils.is_main_process():
            recorder.record_pt_train(loss)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        timer.toc()
        left_time = timer.average_time * (args.pt_epochs - epoch)
        message = "Step1: pretrain now loss is: {} left time is : {} now is: {}".format(loss, timer.format(left_time), datetime.now())
        print(message)
        if utils.is_main_process():
            recorder.record_message('a', message)
        # if (epoch + 1) % args.pt_save_freq == 0:
        #     checkpoint_name = "%04d_%s" % (epoch + 1, "doata_ssl.pth")
        #     filename = os.path.join(args.dota_output, checkpoint_name)
        state = {
        'opt': args,
        'state_dict_encoder': model_without_ddp.encoder.state_dict(),
        'state_dict_head': model_without_ddp.id_head.state_dict(),
        'contrast': contrast.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        }
        save_checkpoint(recorder, args, state, epoch)

    # checkpoint_name = "%04d_%s" % (epoch + 1, "doata_ssl.pth")
    # filename = os.path.join(args.dota_output, checkpoint_name)   
    state = {
            'opt': args,
            'state_dict_encoder': model_without_ddp.encoder.state_dict(),
            'state_dict_head': model_without_ddp.id_head.state_dict(),
            'contrast': contrast.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            }
    save_checkpoint(recorder, args, state, epoch)
    print("finished dota fine-tuning, the trained model is record in: {}".format(args.dota_output))
    return args.dota_output


def save_checkpoint(recorder, args, state, epoch):
    if utils.is_main_process():
        recorder.save_pt_model(args, state, epoch)
        # torch.save(state, filename)
        # del state
        # torch.cuda.empty_cache()


def train_moco(device, epoch, train_loader, model, model_ema, contrast, criterion, optimizer, opt, recorder):
    """
    one epoch training for instance discrimination
    """
    print("==> (MoCo) training...")
    model.train()
    model_ema.eval()

    def set_bn_train(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.train()

    model_ema.apply(set_bn_train)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    prob_meter = AverageMeter()

    end = time.time()
    for idx, (inputs, _, index) in enumerate(train_loader):
        data_time.update(time.time() - end)
        # print(inputs[0].size())
        bsz = inputs[0].size(0)
        # fixed args.batch_size
        if bsz < opt.pt_batch_size:
            print("batch less than 16, continue")
            continue
        inputs[0] = inputs[0].float()
        inputs[1] = inputs[1].float()
        inputs[2] = inputs[2].float()
        inputs[0] = inputs[0].to(device)
        inputs[1] = inputs[1].to(device)
        inputs[2] = inputs[2].to(device)
        index = index.cuda(non_blocking=True)

        # ===================forward=====================
        anchor, positive, negative = inputs

        # here a series of data augmentation
        # ====================================================postive operation=======================
        shuffle_ids, reverse_ids = get_shuffle_ids(bsz)
        feat_q, _ = model(anchor)
        feat_k, _ = model_ema(positive)
        # with torch.no_grad():
        #     positive = positive[shuffle_ids]
        #     feat_k = model_ema(positive)
        #     feat_k = feat_k[reverse_ids]
        feat_n, _ = model(negative)
        out = contrast(feat_q, feat_k, feat_n, index)
        contrast_loss = criterion(out)
        loss = contrast_loss
        prob = out[:, 0].mean()

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        loss_meter.update(loss.item(), bsz)
        prob_meter.update(prob.item(), bsz)

        moment_update(model, model_ema, opt.pt_alpha)
        
        # Waits for all kernels in all streams on a CUDA device to complete
        torch.cuda.synchronize()

        batch_time.update(time.time() - end)
        end = time.time()
        message = ('MoCo Train: [{0}][{1}/{2}]\t'
                   'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                   'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                   'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                   'prob {prob.val:.3f} ({prob.avg:.3f})'.format(
            epoch, idx + 1, len(train_loader), batch_time=batch_time,
            data_time=data_time, loss=loss_meter, prob=prob_meter))
        # print info
        if (idx + 1) % opt.pt_print_freq == 0:
            print(message)
            if utils.is_main_process():
                recorder.record_message('a', message)
                # print(out.shape)
                sys.stdout.flush()
    return loss_meter.avg, prob_meter.avg


def train_dsm_triplet(device, epoch, train_loader, model, optimizer, opt, pos_aug, neg_aug, recorder):
    """
    one epoch training for instance discrimination
    """
    print("==> (DSM triplet) training...")
    model.train()

    def set_bn_train(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    triplet_loss = nn.TripletMarginLoss(margin=0.5, p=2)
    end = time.time()
    for idx, (inputs, _, index) in enumerate(train_loader):
        data_time.update(time.time() - end)

        bsz = inputs[0].size(0)
        # fixed args.batch_size
        if bsz < opt.pt_batch_size:
            print("batch less than 16, continue")
            continue
        for i in range(len(inputs)):
            inputs[i] = inputs[i].float()
            inputs[i] = inputs[i].to(device)
        # ===================forward=====================
        anchor_old, positive, negative = inputs

        # here a series of data augmentation
        # ====================================================postive operation=======================
        anchor = pos_aug(anchor_old)

        
        feat_q = model(anchor)
        feat_k = model(positive)
        feat_n = model(negative)
        intra_loss = triplet_loss(feat_q, feat_k, feat_n)
        inter_loss = triplet_loss(feat_q, feat_k, flip(feat_n, 0))
        # for j in range(bsz-2):
        #     inter_loss += triplet_loss(feat_q, feat_k, shift(feat_n, 0))
        alpha_1 = 1
        alpha_2 = 1
        loss = alpha_1 * intra_loss + alpha_2 * inter_loss
        # print(loss)
        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        loss_meter.update(loss.item(), bsz)
        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()
        message = ('DSM triplet Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=loss_meter))
        # print info
        if (idx + 1) % opt.pt_print_freq == 0:
            print(message)
            if utils.is_main_process():
                recorder.record_message('a', message)
            # print(out.shape)
            sys.stdout.flush()
    return loss_meter.avg


def train_dsm(device, epoch, train_loader, model, model_ema, contrast, criterion, optimizer, opt, pos_aug, neg_aug, recorder):
    """
    one epoch training for instance discrimination
    """
    print("==> (DSM) training...")
    model.train()
    model_ema.eval()

    def set_bn_train(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.train()

    model_ema.apply(set_bn_train)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    prob_meter = AverageMeter()

    end = time.time()
    for idx, (inputs, _, index) in enumerate(train_loader):
        data_time.update(time.time() - end)
        # print(inputs[0].size())
        bsz = inputs[0].size(0)
        # fixed args.batch_size
        # print("batch size {}".format(bsz))
        if bsz < opt.pt_batch_size:
            print("batch less than 16, continue")
            continue
        inputs[0] = inputs[0].float()
        inputs[1] = inputs[1].float()
        inputs[2] = inputs[2].float()
        inputs[0] = inputs[0].to(device)
        inputs[1] = inputs[1].to(device)
        inputs[2] = inputs[2].to(device)

        # what is this?
        index = index.cuda(non_blocking=True)

        # ===================forward=====================
        anchor_old, positive, negative = inputs
        # print(anchor_old.size())
        # here a series of data augmentation
        # ====================================================postive operation=======================
        anchor = pos_aug(anchor_old)
        # print(anchor.shape)
        feat_q = model(anchor)
        feat_k = model_ema(positive)

        feat_n = model(negative)
        
        out = contrast(feat_q, feat_k, feat_n, index)

        # print(out)
        contrast_loss = criterion(out)
        loss = contrast_loss # + tcr_loss  # + sample_loss # + contrast_loss2 # + cls_loss + mixup_loss
        # print(contrast_loss, tcr_loss)
        prob = out[:, 0].mean()

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        loss_meter.update(loss.item(), bsz)
        prob_meter.update(prob.item(), bsz)

        moment_update(model, model_ema, opt.pt_alpha)
        
        #check this 
        torch.cuda.synchronize()

        batch_time.update(time.time() - end)
        end = time.time()
        message = ('DSM Train: [{0}][{1}/{2}]\t'
                   'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                   'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                   'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                   'prob {prob.val:.3f} ({prob.avg:.3f})'.format(
            epoch, idx + 1, len(train_loader), batch_time=batch_time,
            data_time=data_time, loss=loss_meter, prob=prob_meter))
        # print info
        if (idx + 1) % opt.pt_print_freq == 0:
            print(message)
            if utils.is_main_process():
                recorder.record_message('a', message)
            # print(out.shape)
            sys.stdout.flush()
    return loss_meter.avg, prob_meter.avg


if __name__ == '__main__':
    pretext_train_dota()