import os
import pandas as pd
from collections import OrderedDict
from glob import glob
import argparse
import json

def get_split_info(base_path,frame_path,annot_path,split_file=None,normal_only=False):
    split_info = []
    frames_dir = os.path.join(base_path, frame_path)
    exist_files = glob(frames_dir+"/*")
    exist_files = [os.path.basename(_) for _ in exist_files]
    # print(frames_dir)
    if split_file != None:
        split_file_dir = os.path.join(base_path, split_file)
        with open(split_file_dir) as f:
            file_list = f.read().splitlines()
        for filename in file_list:
            if filename not in exist_files:
                continue 
            annot_file = os.path.join(base_path,annot_path,filename+".json")
            with open(annot_file) as f:
                annot = json.load(f)
            if normal_only:         
                video_length = annot['anomaly_start']
                if video_length ==0:
                    continue
            else:
                video_length = annot['num_frames']
            accident_name = annot['accident_name']
            ego_involve = annot['ego_involve']
            
            split_info.append([filename, video_length, accident_name, ego_involve])
    else:
        for filename in exist_files:
            annot_file = os.path.join(base_path, annot_path,filename+".json")
            with open(annot_file) as f:
                annot = json.load(f)
            if normal_only:         
                video_length = annot['anomaly_start']
                if video_length ==0:
                    continue
            else:
                video_length = annot['num_frames']
            accident_name = annot['accident_name']
            ego_involve = annot['ego_involve']
            
            split_info.append([filename, video_length, accident_name, ego_involve])
            
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
    split_info_tr = get_split_info(args.base_path, args.frame_path, args.annot_path, args.train_split_file,args.normal)
    write_split_files(setting_path, split_id, "train", split_info_tr)
    #val
    split_info_val = get_split_info(args.base_path, args.frame_path, args.annot_path, args.val_split_file,args.normal)
    write_split_files(setting_path, split_id, "val", split_info_val)
    #all
    split_info_all = get_split_info(args.base_path, args.frame_path, args.annot_path,None,args.normal)
    write_split_files(setting_path, split_id, "all", split_info_all)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate split files for DOTA')
    parser.add_argument('--settings', default='./datasets/settings/dota_seq',
                        help='path to dataset setting files')
    parser.add_argument('--base-path', default='.',
                        help='path to dataset files')  
    parser.add_argument('--frame-path', default='frames',
                        help='path to image files') 
    parser.add_argument('--annot-path', default='annotations',
                        help='path to annotation files') 
    parser.add_argument('--train-split-file', default='train_split.txt',
                        help='train split file')
    parser.add_argument('--val-split-file', default='val_split.txt',
                        help='validation split file')  
    parser.add_argument('--split-id', default=1, type=int, metavar='S',
                    help='id of the split files, e.g. train{split-id}.txt')  
    parser.add_argument('--normal', action='store_true',
                    help='only select normal parts')  

    args = parser.parse_args()
    main(args)