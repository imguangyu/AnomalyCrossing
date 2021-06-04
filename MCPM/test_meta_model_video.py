import torch 
import torch.nn as nn
import numpy as np
# from datasets import dota_ctx
from sklearn import metrics
import argparse
import os
from models.baselinefinetune import BaselineFinetune

DOTA_CLASSES = ['lateral','leave_to_left','leave_to_right',
               'moving_ahead_or_waiting','obstacle','oncoming',
               'pedestrian','start_stop_or_stationary','turning',
               'unknown']

def finetune(args, support_dataset, query_dataset, device):
    # each element in the datasets is an arracy of save node features
    # with dim [num_points, feat_dim(512)]
    
    batch_size = args.finetune_batch_size
    num_epochs = args.finetune_epochs
    
    model = BaselineFinetune(device=device, 
                            n_way=args.way, 
                            n_support=args.shot,
                            cls_head_type=args.cls_head_type,
                            batch_size=args.finetune_batch_size,
                            num_epochs=args.finetune_epochs,
                            threshold=args.test_th,
                            temperature=args.temperature,
                            fc_inter_dim=args.fc_inter_dim
                            )
    if args.cls_head_type in ['fc', 'fc2', 'cosine_dist']:
        adaptation = True
    else:
        adaptation = False
    if adaptation:    
        eval_acc, train_acc, pred_eval, target_query = model.set_forward_adaptation_video(support_dataset, 
                                                                            query_dataset, 
                                                                            mode=args.fuse_type)
    else:
        eval_acc, train_acc, pred_eval, target_query = model.set_forward_video(support_dataset, 
                                                                query_dataset, 
                                                                mode=args.fuse_type)

    return train_acc, eval_acc, pred_eval, target_query, model


def eveluate(args):
    if args.dataset == "dota":
        from datasets import dota_ctx as data_ctx
    elif args.dataset == "ucf_crime":
        from datasets import ucf_crime_ctx as data_ctx
    else:
        print("Unknown dataset!!!")
    modality='rgb'

    best_acc = {'normal':-1,'abnormal':-1,'train':-1,'mixed':-1,'overall':-1}
    save_path = os.path.join(args.save_path, args.cls_head_type)

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
    
    if args.selected_cls:
        selected_cls = args.selected_cls
    else:
        selected_cls = None 
    
    if args.ego_envolve:
        ego_envolve = args.ego_envolve
    else:
        ego_envolve = None 
 
    print("selected_cls: {}; ego_envolve: {}; cls_head: {}".format(selected_cls, ego_envolve, args.cls_head_type))
    fs_worset = data_ctx.Workset(root=args.ctx_path, 
                      source=split_file, 
                      n_batch=args.iter_num, 
                      n_cls=args.way, 
                      n_shot=args.shot,
                      n_query=args.query, 
                      ego_envolve=ego_envolve, 
                      selected_cls=selected_cls,
                      min_duration=args.min_duration,
                      max_duration=args.max_duration)

    acc_all = []
    for i in range(len(fs_worset)):
        clips_set, labels_set = fs_worset[i]
        support_dataset = data_ctx.ctx_set(clips_set, 
                                     args.way,
                                     args.shot,
                                     args.query,
                                     'train',
                                     args.num_augs,
                                     args.aug_folder,
                                     args.name_pattern_ctx_aug,
                                     ego_envolve, 
                                     selected_cls, 
                                     args.name_pattern_ctx,
                                     )
        query_dataset = data_ctx.ctx_set(clips_set, 
                                     args.way,
                                     args.shot,
                                     args.query,
                                     'eval',
                                     0,
                                     None,
                                     None,
                                     ego_envolve, 
                                     selected_cls, 
                                     args.name_pattern_ctx,
                                     )

        train_acc, eval_acc, pred_eval, query_label, model = finetune(args, support_dataset, query_dataset, device)
        acc_all.append(eval_acc)
        confusion_table = metrics.confusion_matrix(pred_eval, query_label)
        recall_table = confusion_table / np.sum(confusion_table, axis=0)
        mixed_acc = eval_acc + 0.1 * train_acc
        
        if mixed_acc > best_acc['mixed']:
            best_acc['mixed'] = mixed_acc
            best_acc['normal'] = recall_table[0,0]
            best_acc['abnormal'] = recall_table[1,1]
            best_acc['train'] = train_acc
            best_acc['overall'] = eval_acc
            # print("Best model recall: normal {:.4f}, abnormal {:.4f}, trainn {:.4f}".format(recall_table[0,0], recall_table[1,1], train_acc))
            if args.save_model:
                torch.save(model.cls_head.state_dict(), os.path.join(save_path, model.cls_head_type+'.pth'))
 
    best_res = "Best model acc: normal {:.4f}, abnormal {:.4f}, overall {:.4f}, train {:.4f}".format(
                best_acc['normal'], 
                best_acc['abnormal'], 
                best_acc['overall'],
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', default='dota',
                    choices=["dota", "ucf_crime"],
                    help='dataset: dota or orthers')
    parser.add_argument('--settings', metavar='DIR', default='./datasets/settings',
                    help='path to dataset setting files')
    parser.add_argument('--ctx-path', metavar='DIR', default='./datasets/dota/frames',
                    help='path to dataset files')    
    parser.add_argument('--phase', default="all", help='phase of the split file, e.g. train, val or all')
    parser.add_argument('--name-pattern-ctx',  default='ctx_%s.npy',
                        help='name pattern of the context vector files')   
    parser.add_argument('-s', '--split', default=1, type=int, metavar='S',
                    help='which split of data to work on (default: 1)')
    parser.add_argument('--device', default='cuda', help='computing device cuda or cpu')
    parser.add_argument('--iter-num', default=100, type=int,  help='number of random tests to be done')
    parser.add_argument('--query', default=15, type=int,  help='number of quries in test')
    parser.add_argument('--way'  , default=5, type=int,  help='class num to classify for testing (validation) ') #baseline and baseline++ only use this parameter in finetuning
    parser.add_argument('--shot', type=int, default=10)
    parser.add_argument('--test-th', default=0.5, type=float,  help='threshold for the test')
    parser.add_argument('--feat-dim', default=512, type=int,  help='dimension of the output features from the encoder')
    parser.add_argument('--finetune-batch-size', default=4, type=int,  help='batch size for training the stgcn')
    parser.add_argument('--finetune-epochs', type=int, default=301, help='epochs for training the stgcn')
    parser.add_argument('--finetune-lr', type=float, default=0.01, help='learning rate for training the stgcn')
    parser.add_argument('--selected-cls', default="",
                    help="Selected test class for dota. Default test on all classes")
    parser.add_argument('--ego-envolve', default="",
                        choices=["True","False"],
                        help='Specify use only ego or non-ego or both. Default test on both')
    parser.add_argument('--save-model'  , action='store_true', help='save the best model or not')
    parser.add_argument('--save-path', type=str, default=None)
    parser.add_argument('--min-duration', type=int, default=20, 
                        help='min duration of the input clips, should match the pre-computed context vectors')
    parser.add_argument('--max-duration', type=int,
                        help='max duration of the input clips, should match the pre-computed context vectors')
    parser.add_argument('--num-augs', type=int, default=0, help='number of augmented samples to be loaded corresponding to 1 original sample')
    parser.add_argument('--num-repeat-org', type=int, default=1, help='number of original sampels (not augmented) will be copied')
    parser.add_argument('--name-pattern-ctx-aug',  default='ctx_%s_%d.npy',
                        help='name pattern of the augmented context vector files')
    parser.add_argument('--aug-folder',  default='augmented1',
                        help='subfolder to holf the augmented ctx vectors')
    parser.add_argument('--fuse-type',  default="mean",  help='method to integrate the clip scores to be the video level score')
    parser.add_argument('--fc-inter-dim', type=int, default=128, help='inter fc layer dim for fc2 head')
    parser.add_argument('--temperature', type=float, default=128)
    parser.add_argument('--cls-head-type', default='protonet', help='loss type of the clf head')

    args = parser.parse_args()

    eveluate(args)

