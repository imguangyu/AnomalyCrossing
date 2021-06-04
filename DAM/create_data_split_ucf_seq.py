import os
import pandas as pd
from collections import OrderedDict
from glob import glob
import argparse
import json

def get_split_info(frame_path):
    split_info = []
    if not os.path.exists(frame_path):
        print("Image files %s for BDD dataset doesn't exist." % (frame_path))
        sys.exit()
    else:
        img_folders = glob(os.path.join(frame_path, '*'))
        for clip_path in img_folders:
            folder_name = os.path.basename(clip_path) 
            if not folder_name.startswith("Normal"):
                continue
            frames = glob(os.path.join(clip_path, '*'))
            video_length = len(frames)
            dirname = os.path.basename(os.path.dirname(clip_path))
            folder_name_ex = os.path.join(dirname, folder_name)
            split_info.append([folder_name_ex, video_length])
            # cls_name = folder_name.split('_')[0]
            # split_info.append([folder_name_ex, video_length, cls_name])
            
    return split_info

def write_split_files(save_path, split_id, phase, split_info):
    # split_id = 1
    # save_path = '.datasets/settings/kinetics100/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    split_file = os.path.join(save_path,"{}_rgb_split{}.txt".format(phase,split_id))

    with open(split_file, 'w') as f:
        for item in split_info:
            item = [str(_) for _ in item]
            f.write("%s\n" % ",".join(item))

def main(args):
    setting_path = args.settings
    split_id = args.split_id
    #train
    if args.train_path:
        split_info_tr = get_split_info(args.train_path)
        write_split_files(setting_path, split_id, "train", split_info_tr)
    #val
    if args.val_path:
        split_info_tr = get_split_info(args.val_path)
        write_split_files(setting_path, split_id, "val", split_info_tr)

    #test
    if args.test_path:
        split_info_tr = get_split_info(args.test_path)
        write_split_files(setting_path, split_id, "test", split_info_tr)
    #all
    # split_info_all = get_split_info(args.base_path, args.frame_path, args.annot_path,None,args.normal)
    # write_split_files(setting_path, split_id, "all", split_info_all)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate split files for BDD')
    parser.add_argument('--settings', default='./datasets/settings/bdd_seq',
                        help='path to dataset setting files')
    parser.add_argument('--train-path', default='',
                        help='path to image files') 
    parser.add_argument('--val-path', default='',
                        help='path to image files') 
    parser.add_argument('--test-path', default='',
                        help='path to image files') 
    parser.add_argument('--split-id', default=1, type=int, metavar='S',
                    help='id of the split files, e.g. train{split-id}.txt')  

    args = parser.parse_args()
    main(args)