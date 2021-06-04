import os
import pandas as pd
from collections import OrderedDict
from glob import glob
import argparse
import json

def get_split_info(base_path,frame_path,annot_path,split_file=None):
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
            annot_file = os.path.join(base_path, annot_path, filename+".json")
            with open(annot_file) as f:
                annot = json.load(f)
            anomaly_start = int(annot['anomaly_start'])
            anomaly_end = int(annot['anomaly_end']) - 1
            accident_name = annot['accident_name']
            ego_involve = annot['ego_involve']
            #anomaly case
            split_info.append([filename, anomaly_start, anomaly_end, 'abnormal', accident_name, ego_involve])
            # normal case
            # part after the accident was discarded because it is hard to say it is normal or abnormal
            if anomaly_start > 0:
                split_info.append([filename, 0, anomaly_start - 1, 'normal', accident_name, ego_involve]) 
    else:
        for filename in exist_files:
            annot_file = os.path.join(base_path, annot_path,filename+".json")
            with open(annot_file) as f:
                annot = json.load(f)
            # note anomaly_start time is exactly the time abnormal starts
            # but anomaly_end time is one more frame after the last abnormal frame
            anomaly_start = int(annot['anomaly_start'])# - 1
            anomaly_end = int(annot['anomaly_end']) - 1
            accident_name = annot['accident_name']
            ego_involve = annot['ego_involve']
            #anomaly case
            split_info.append([filename, anomaly_start, anomaly_end, 'abnormal', accident_name, ego_involve])
            # normal case
            # part after the accident was discarded because it is hard to say it is normal or abnormal
            if anomaly_start > 0:
                split_info.append([filename, 0, anomaly_start - 1, 'normal', accident_name, ego_involve])
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
    split_info_tr = get_split_info(args.base_path, args.frame_path, args.annot_path, args.train_split_file)
    write_split_files(setting_path, split_id, "train", split_info_tr)
    #val
    split_info_val = get_split_info(args.base_path, args.frame_path, args.annot_path, args.val_split_file)
    write_split_files(setting_path, split_id, "val", split_info_val)
    #all
    split_info_all = get_split_info(args.base_path, args.frame_path, args.annot_path)
    write_split_files(setting_path, split_id, "all", split_info_all)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate split files for DOTA')
    parser.add_argument('--settings', default='./datasets/settings/dota',
                        help='path to dataset setting files')
    parser.add_argument('--base-path', default='./datasets/DOTA',
                        help='path to dataset files')  
    parser.add_argument('--frame-path', default='base-path/frames',
                        help='path to image files') 
    parser.add_argument('--annot-path', default='base-path/annotations',
                        help='path to annotation files') 
    parser.add_argument('--train-split-file', default='train_split.txt',
                        help='train split file')
    parser.add_argument('--val-split-file', default='val_split.txt',
                        help='validation split file')  
    parser.add_argument('--split-id', default=1, type=int, metavar='S',
                    help='id of the split files, e.g. train{split-id}.txt')  

    args = parser.parse_args()
    main(args)