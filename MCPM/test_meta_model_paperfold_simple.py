import torch 
import numpy as np
from models.baselinefinetune import BaselineFinetune
import datasets.feature_loader as feat_loader
from sklearn import metrics
import argparse
import os

def feature_evaluation(cl_data_file, 
                       model, 
                       save_path,
                       args):
    n_way=args.test_n_way
    n_support=args.test_n_shot
    n_query=args.test_n_query
    save_best_model_features=args.save_features
    save_best_model=args.save_best_model
    test_mode = args.test_mode
    
    # LP params
    k_lp = args.k_lp 
    delta = args.delta
    alpha = args.alpha

    if args.cls_head_type in ['fc', 'cosine_dist']:
        adaptation = True
    else:
        adaptation = False

    class_list = list(cl_data_file.keys())
    # print(class_list)
    assert n_way <= len(class_list), "Num of ways should be <= num of classes"
    if n_way < len(class_list):
        select_class = random.sample(class_list,n_way)
    else:
        select_class = list(class_list) #0 is normal and 1 is abnormal
    
    # n_query < 0 means using all the left samples as queries
    if n_query > 0:
        z_support = []
        z_query = []
        query_label = []
        support_label = []
        for cl in select_class:
            img_feat = cl_data_file[cl]
            perm_ids = np.random.permutation(len(img_feat)).tolist()
            z_support += [np.squeeze(img_feat[perm_ids[i]]) for i in range(n_support)] 
            z_query += [ np.squeeze( img_feat[perm_ids[i]]) for i in range(n_support, n_support + n_query) ] 
            query_label += [cl] * n_query
            support_label += [cl] * n_support
    else:
        z_support = []
        z_query = []
        query_label = []
        support_label = []
        for cl in select_class:
            img_feat = cl_data_file[cl]
            perm_ids = np.random.permutation(len(img_feat)).tolist()
            z_support += [np.squeeze( img_feat[perm_ids[i]]) for i in range(n_support)]    # stack each batch
            z_query += [np.squeeze( img_feat[perm_ids[i]]) for i in range(n_support, len(img_feat)) ] 
            query_label += [cl] * (len(img_feat) - n_support)
            support_label += [cl] * n_support

    z_support = torch.from_numpy(np.array(z_support))
    z_query = torch.from_numpy(np.array(z_query))
    query_label = torch.from_numpy(np.array(query_label)) 
    support_label = torch.from_numpy(np.array(support_label))
    z_all = {'z_support':z_support, 
              'z_query':z_query,
              'support_labels':support_label,
              'query_labels':query_label
            }
        
    model.n_query = n_query
    if adaptation:
        if test_mode == "lp":
            eval_acc, train_acc, pred_eval = model.set_forward_adaptation_lp(z_all, 
                                              is_feature=True, 
                                              k_lp=k_lp, 
                                              delta=delta,
                                              alpha=alpha)
        else:
            eval_acc, train_acc, pred_eval = model.set_forward_adaptation(z_all, is_feature=True)
    else:
        if test_mode == "lp":
            eval_acc, train_acc, pred_eval = model.set_forward_lp(z_all, 
                                              is_feature=True, 
                                              k_lp=k_lp, 
                                              delta=delta,
                                              alpha=alpha)
        else:
            eval_acc, train_acc, pred_eval = model.set_forward(z_all, is_feature=True)
    
    confusion_table = metrics.confusion_matrix(pred_eval, query_label)
    recall_table = confusion_table / np.sum(confusion_table, axis=0)
    # print("nomal acc: {:.4f}; abnormal acc: {:.4f}".format(recall_table[0,0], recall_table[1,1]))
    mixed_acc = eval_acc + 0.1 * train_acc

    if mixed_acc > best_acc['mixed']:
        best_acc['mixed'] = mixed_acc
        best_acc['normal'] = recall_table[0,0]
        best_acc['abnormal'] = recall_table[1,1]
        best_acc['train'] = train_acc
        # print("Best model recall: normal {:.4f}, abnormal {:.4f}, trainn {:.4f}".format(recall_table[0,0], recall_table[1,1], train_acc))
        if adaptation and save_best_model:
            torch.save(model.cls_head.state_dict(), os.path.join(save_path, model.cls_head_type+'.pth'))
        if save_best_model_features:
            with open(os.path.join(save_path, 'features_best_model_{}.pickle'.format(model.cls_head_type)), 'wb') as f:
                pickle.dump(z_all, f)
    return eval_acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--save-path', type=str, default=None)
    parser.add_argument('--device', default='cuda', help='computing device cuda or cpu')
    #test_setting
    parser.add_argument('--feature-file-train', required=True, help ='dir for saved features for training')
    parser.add_argument('--feature-file-eval', required=True, help ='dir for saved features for evaluation')
    parser.add_argument('--cls-head-type', default='fc', help='loss type of the clf head')
    parser.add_argument('--test-th'      , default=0.5, type=float,  help='threshold for the test')
    # parser.add_argument('--adaptation'  , action='store_true', help='further adaptation in test time or not')
    parser.add_argument('--finetune-batch-size'      , default=4, type=int,  help='batch size for the clf head')
    parser.add_argument('--finetune-epochs', type=int, default=301)
    parser.add_argument('--temperature', type=float, default=128)
    parser.add_argument('--test-mode', default="vanila",
                        choices=["vanila","lp"],
                        help='Specify use only ego or non-ego or both. Default test on both')
    parser.add_argument('--k_lp', type=int, default=10, help='num of nearest beibors in LP')
    parser.add_argument('--delta', type=float, default=0.5, help='delta in LP')
    parser.add_argument('--alpha', type=float, default=0.5, help='alpha in LP')
    parser.add_argument('--save-best-model'  , action='store_true', help='save the best model or not')

    args = parser.parse_args()
    best_acc = {'normal':-1,'abnormal':-1,'train':-1,'mixed':-1}
    save_path = os.path.join(args.save_path,args.cls_head_type)
    cl_data_file_train = feat_loader.init_loader(args.feature_file_train)
    cl_data_file_eval = feat_loader.init_loader(args.feature_file_eval)

    # print("evaluating over %d examples"%(args.test_n_query))
    device = torch.device(args.device)
   
    model = BaselineFinetune(device=device, 
                            n_way=args.test_n_way, 
                            n_support=args.test_n_shot,
                            cls_head_type=args.cls_head_type,
                            batch_size=args.finetune_batch_size,
                            num_epochs=args.finetune_epochs,
                            threshold=args.test_th,
                            temperature=args.temperature
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
    for i in range(args.iter_num):
        acc = feature_evaluation(cl_data_file, model, save_path, args)
        acc_all.append(acc)

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
                                         args.test_n_way,
                                         args.test_n_shot,
                                         args.test_n_query
                                         ))
    info = "Mean acc: {:.4f}, Std: {:.4f}, Max acc: {:.4f}, Min acc: {:.4f}".format(
            np.mean(acc_all), np.std(acc_all), np.max(acc_all), np.min(acc_all)
    )
    settings = "{:d} way, {:d} shot, {:d} query, Num iters: {:d}".format(
          args.test_n_way, 
          args.test_n_shot, 
          args.test_n_query,
          args.iter_num)
    print(settings)
    print(info)
    print(best_res)
    print()
    with open(log_file, 'w') as f:
        f.write("{}\n".format(settings))
        f.write("{}\n".format(info))
        f.write(best_res)