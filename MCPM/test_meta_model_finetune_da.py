import torch 
import torch.backends.cudnn as cudnn
import numpy as np
import models
from models.baselinefinetune import BaselineFinetune_DA
from datasets import dota_da
from sklearn import metrics
import argparse
import os

def finetune(args):
        # default scale
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
    
    #overwrite the default scale
    if args.scale:
        scale = args.scale

    input_size = int(224 * scale)
    width = int(340 * scale)
    height = int(256 * scale)

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

    print('length %d, img size %d' %(length, input_size))
    # Data transforming
    if modality == "rgb" or modality == "pose":
        is_color = True
        scale_ratios = [1.0, 0.875, 0.75, 0.66]
        if 'I3D' in args.arch:
            if 'resnet' in args.arch:
                clip_mean = [0.45, 0.45, 0.45] 
                clip_std = [0.225, 0.225, 0.225] 
            else:
                clip_mean = [0.5, 0.5, 0.5] 
                clip_std = [0.5, 0.5, 0.5] 
            #clip_std = [0.25, 0.25, 0.25] * args.num_seg * length
        elif 'MFNET3D' in args.arch:
            clip_mean = [0.48627451, 0.45882353, 0.40784314]
            clip_std = [0.234, 0.234, 0.234]
        elif "3D" in args.arch:
            clip_mean = [114.7748, 107.7354, 99.4750] 
            clip_std = [1, 1, 1] 
        elif "r2plus1d" in args.arch:
            clip_mean = [0.43216, 0.394666, 0.37645]
            clip_std = [0.22803, 0.22145, 0.216989] 
        elif "rep_flow" in args.arch:
            clip_mean = [0.5, 0.5, 0.5] 
            clip_std = [0.5, 0.5, 0.5]  
        elif "slowfast" in args.arch:
            clip_mean = [0.45, 0.45, 0.45] 
            clip_std = [0.225, 0.225, 0.225] 
        else:
            clip_mean = [0.485, 0.456, 0.406] 
            clip_std = [0.229, 0.224, 0.225] 
    elif modality == "pose":
        is_color = True
        scale_ratios = [1.0, 0.875, 0.75, 0.66]
        clip_mean = [0.485, 0.456, 0.406] 
        clip_std = [0.229, 0.224, 0.225] 
    elif modality == "flow":
        is_color = False
        scale_ratios = [1.0, 0.875, 0.75, 0.66]
        if 'I3D' in args.arch:
            clip_mean = [0.5, 0.5] 
            clip_std = [0.5, 0.5] 
        elif "3D" in args.arch:
            clip_mean = [127.5, 127.5] 
            clip_std = [1, 1]       
        else:
            clip_mean = [0.5, 0.5]
            clip_std = [0.226, 0.226] 
    elif modality == "both":
        is_color = True
        scale_ratios = [1.0, 0.875, 0.75, 0.66]
        clip_mean = [0.485, 0.456, 0.406, 0.5, 0.5]
        clip_std = [0.229, 0.224, 0.225, 0.226, 0.226] 
    else:
        print("No such modality. Only rgb and flow supported.")

    best_acc = {'normal':-1,'abnormal':-1,'train':-1,'mixed':-1}
    save_path = os.path.join(args.save_path,args.cls_head_type)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    if args.selected_cls:
        selected_cls = args.selected_cls
    else:
        selected_cls = None 
    
    if args.ego_envolve:
        ego_envolve = args.ego_envolve
    else:
        ego_envolve = None 

    # print("evaluating over %d examples"%(args.test_n_query))
    device = torch.device(args.device)

    setting_file = "%s_%s_split%d.txt" % (args.phase, modality, args.split)
    split_file = os.path.join(args.settings, args.dataset, setting_file)

    normalize_param=dict(mean=clip_mean, std=clip_std)
    jitter_param = dict(brightness=0.2, contrast=0.2, 
                        saturation=0.2, hue=0.2, 
                        consistent=True, p=0.5)
    scale_param=dict(size=(width,height))
    guassian_blur_param=dict(kernel_size=3, sigma=0.2, p=0.5)
    rand_rotation_param = dict(consistent=True, degree=10, p=0.5)

    datamgr = dota_da.SetDataManager(frames_path=args.frames_path,
                                split_file=split_file,
                                image_size=input_size,
                                n_way=args.way,
                                n_support=args.shot,
                                n_query=args.query,
                                n_eposide=args.iter_num,
                                normalize_param=normalize_param,
                                jitter_param=jitter_param,
                                scale_param=scale_param,
                                guassian_blur_param=guassian_blur_param,
                                rand_rotation_param=rand_rotation_param,
                                phase='val',
                                modality=modality,
                                ego_envolve=ego_envolve,
                                selected_cls=selected_cls,
                                name_pattern_rgb=args.name_pattern_rgb,
                                is_color=is_color,
                                length=length,
                                num_seg=args.num_seg
                                )
    data_loader = datamgr.get_data_loader(args.num_aug, args.num_workers)

    model = BaselineFinetune_DA(device=device, 
                         cls_head_type=args.cls_head_type,
                         arch=args.arch,
                         modelLocation=args.modelLocation,
                         saved_model_name=args.saved_model_name,
                         length=length,
                         image_size=input_size,
                         feat_dim=args.feat_dim,
                         n_way=args.way, 
                         n_support=args.shot,
                         batch_size=args.finetune_batch_size,
                         num_epochs=args.finetune_epochs,
                         threshold=args.test_th,
                         freeze_backbone=args.freeze_backbone,
                         train_full_encoder=args.train_full_encoder,
                         fc_inter_dim=args.fc_inter_dim
                            )

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    if args.selected_cls:
        selected_cls = args.selected_cls
    else:
        selected_cls = None 
    
    if args.ego_envolve:
        ego_envolve = args.ego_envolve
    else:
        ego_envolve = None 
        
    if args.test_mode == "lp":
        print("selected_cls: {}; ego_envolve: {}; cls_head: {}; delta: {}; k_lp: {}; alpha: {}".format(
          selected_cls, 
          ego_envolve, 
          args.cls_head_type, 
          args.delta, args.k_lp, args.alpha)) 
    else:
        print("selected_cls: {}; ego_envolve: {}; cls_head: {}".format(selected_cls, ego_envolve, args.cls_head_type))
    
    acc_all = []
    for i, (inputs) in enumerate(data_loader):
        if args.test_mode == 'lp':
            eval_acc, train_acc, pred_eval, query_label = model.set_forward_adaptation(inputs, 
                                                                                       do_lp=True,
                                                                                       k_lp=args.k_lp,
                                                                                       delta=args.delta,
                                                                                       alpha=args.alpha
                                                                                       )
        else:
            eval_acc, train_acc, pred_eval, query_label = model.set_forward_adaptation(inputs)
        acc_all.append(eval_acc)
        confusion_table = metrics.confusion_matrix(pred_eval, query_label)
        recall_table = confusion_table / np.sum(confusion_table, axis=0)
        mixed_acc = eval_acc + 0.1 * train_acc

        if mixed_acc > best_acc['mixed']:
            best_acc['mixed'] = mixed_acc
            best_acc['normal'] = recall_table[0,0]
            best_acc['abnormal'] = recall_table[1,1]
            best_acc['train'] = train_acc
            # print("Best model recall: normal {:.4f}, abnormal {:.4f}, trainn {:.4f}".format(recall_table[0,0], recall_table[1,1], train_acc))
            if args.save_model:
                torch.save({'encoder': model.encoder_without_dp.state_dict(), 'cls_head': model.cls_head.state_dict()}, os.path.join(save_path, model.cls_head_type + '_combined.pth'))
 
        # print("{} steps reached Best model recall: normal {:.4f}, abnormal {:.4f}, overall {:.4f}, train {:.4f}".format(
        #   i, 
        #   best_acc['normal'], 
        #   best_acc['abnormal'], 
        #   0.5*best_acc['normal'] + 0.5*best_acc['abnormal'],
        #   best_acc['train']
        #   ))

    best_res = "Best model acc: normal {:.4f}, abnormal {:.4f}, overall {:.4f}, train {:.4f}".format(
                best_acc['normal'], 
                best_acc['abnormal'], 
                0.5*best_acc['normal'] + 0.5*best_acc['abnormal'],
                best_acc['train'])
    
    log_file = os.path.join(save_path, 
              "{}_{}iters_{}w{}s{}q_log.txt".format(args.cls_head_type, 
                                         args.iter_num, 
                                         args.way,
                                         args.shot,
                                         args.query
                                         ))
    info = "Mean acc: {:.4f}, Std: {:.4f}, Max acc: {:.4f}, Min acc: {:.4f}".format(
            np.mean(acc_all), np.std(acc_all), np.max(acc_all), np.min(acc_all)
    )
    settings = "{:d} way, {:d} shot, {:d} query, Num iters: {:d}".format(
          args.way, 
          args.shot, 
          args.query,
          args.iter_num)
    print(settings)
    print(info)
    print(best_res)
    print()
    with open(log_file, 'w') as f:
        f.write("{}\n".format(settings))
        f.write("{}\n".format(info))
        f.write(best_res)

DOTA_CLASSES = ['lateral','leave_to_left','leave_to_right',
               'moving_ahead_or_waiting','obstacle','oncoming',
               'pedestrian','start_stop_or_stationary','turning',
               'unknown']

model_names = sorted(name for name in models.__dict__
    if not name.startswith("__")
    and callable(models.__dict__[name]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', default='dota',
                    choices=["dota"],
                    help='dataset: dota')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='rgb_r2plus1d_32f_34_encoder',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: rgb_resneXt3D64f101)')
    parser.add_argument('--settings', metavar='DIR', default='./datasets/settings',
                    help='path to dataset setting files')
    parser.add_argument('--frames-path', metavar='DIR', default='./datasets/dota/frames',
                    help='path to dataset files')    
    parser.add_argument('--phase', default="all", help='phase of the split file, e.g. train, val or all')
    parser.add_argument('--name-pattern-rgb',  default='%06d.jpg',
                        help='name pattern of the frame files') 
    parser.add_argument('--num-seg', default=1, type=int,
                    metavar='N', help='Number of segments for temporal LSTM (default: 16)')  
    parser.add_argument('-s', '--split', default=1, type=int, metavar='S',
                    help='which split of data to work on (default: 1)')
    
    parser.add_argument('--device', default='cuda', help='computing device cuda or cpu')
    parser.add_argument('--modelLocation', default="", help='path of the saved model')
    parser.add_argument('--saved-model-name', default="model_best", help='name of the saved model')

    parser.add_argument('--cls-head-type', default='protonet', help='loss type of the clf head')
    parser.add_argument('--iter-num', default=100, type=int,  help='number of random tests to be done')
    parser.add_argument('--query', default=15, type=int,  help='number of quries in test')
    parser.add_argument('--way'  , default=5, type=int,  help='class num to classify for testing (validation) ') #baseline and baseline++ only use this parameter in finetuning
    parser.add_argument('--shot', type=int, default=10)
    parser.add_argument('--test-th', default=0.5, type=float,  help='threshold for the test')
    parser.add_argument('--feat-dim', default=512, type=int,  help='dimension of the output features from the encoder')
    parser.add_argument('--finetune-batch-size'      , default=4, type=int,  help='batch size for the clf head')
    parser.add_argument('--finetune-epochs', type=int, default=301)
    parser.add_argument('--num-aug', type=int, default=10, help='Number of augmented datasets')
    parser.add_argument('--num-workers', type=int, default=8, help='Number of workers for the data loader')
    # parser.add_argument('--temperature', type=float, default=128)
    # parser.add_argument('--save-features'  , action='store_true', help='save the best models features')
    parser.add_argument('--selected-cls', default="",
                    help="Selected test class for dota. Default test on all classes")
    parser.add_argument('--ego-envolve', default="",
                        choices=["True","False"],
                        help='Specify use only ego or non-ego or both. Default test on both')
    parser.add_argument('--scale', default=None, type=float,
                       help='customized image size scale')
    parser.add_argument('--test-mode', default="vanila",
                        choices=["vanila","lp"],
                        help='Specify use only ego or non-ego or both. Default test on both')
    parser.add_argument('--k_lp', type=int, default=10, help='num of nearest beibors in LP')
    parser.add_argument('--delta', type=float, default=0.5, help='delta in LP')
    parser.add_argument('--alpha', type=float, default=0.5, help='alpha in LP')
    parser.add_argument('--freeze-backbone'  , action='store_true', help='fine tune the encoder')
    parser.add_argument('--train-full-encoder'  , action='store_true', help='train the whole encoder otherwise onlhy train the last block')
    parser.add_argument('--save-model'  , action='store_true', help='save the best model or not')
    parser.add_argument('--save-path', type=str, default=None)
    parser.add_argument('--fc-inter-dim', type=int, default=128, help='inter fc layer dim for fc2 head')
    args = parser.parse_args()

    finetune(args)

