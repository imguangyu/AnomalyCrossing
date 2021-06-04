import os
import pandas as pd
from collections import OrderedDict
from glob import glob
import argparse
import json

def get_split_info(frame_path, annot_file):
    with open(annot_file) as f:
        annot_info = f.read().splitlines()
    
    split_info = []
    for item in annot_info:
        annot = item.split(" ")
        annot = [_.strip() for _ in annot if _.strip() != ""]
        #remove the file ext
        vid_name = annot[0].split(".")[0]
        # class name
        cls_name = annot[1]
        #get the image folder name. name convention is {cls}_xxxx
        vid_name = vid_name.replace(cls_name, cls_name + "_")
        #get the video length
        video_len = len(glob("{}/*".format(os.path.join(frame_path, vid_name))))
        if cls_name == "Normal":
            split_info.append([vid_name, 1, video_len, cls_name])
        else:
            #the images are extracted from videos with fps 10 but the orifinal fps is 30
            #so the temporal annots are divided by 3
            #event1
            start_time_1 = max(int(annot[2]) // 3, 1)
            end_time_1 = min(int(annot[3]) // 3, video_len)
            if start_time_1 > 0 and  end_time_1 > 0 and end_time_1 > start_time_1:
                split_info.append([vid_name, start_time_1, end_time_1, cls_name])
            #event2
            start_time_2 = max(int(annot[4]) // 3, 1)
            end_time_2 = min(int(annot[5]) // 3, video_len)
            if start_time_2 > 0 and  end_time_2 > 0 and end_time_2 > start_time_2:
                split_info.append([vid_name, start_time_2, end_time_2, cls_name])

    return split_info

#for the corning created temporal annotations
def get_split_info_corning(frame_path, annot_file, image_fps):
    with open(annot_file) as f:
        annot_info = f.read().splitlines()
    
    split_info = []
    for item in annot_info:
        annot = item.split(" ")
        annot = [_.strip() for _ in annot if _.strip() != ""]
        #remove the file ext
        vid_name = annot[0].split(".")[0]
        # class name
        cls_name = annot[1]
        #get the image folder name. name convention is {cls}_xxxx
        vid_name = vid_name.replace(cls_name, cls_name + "_")
        #get the video length
        video_len = len(glob("{}/*".format(os.path.join(frame_path, vid_name))))
        video_fps = float(annot[4])
        fps_ratio = video_fps / image_fps
        if cls_name == "Normal":
            split_info.append([vid_name, 1, video_len, cls_name])
        else:
            #the images are extracted from videos with fps 10 but the orifinal fps is 30
            #so the temporal annots are divided by fps_ratio
            start_time = max(int(int(annot[2]) / fps_ratio), 1)
            end_time = min(int(int(annot[3]) / fps_ratio), video_len)
            if end_time > start_time:
                split_info.append([vid_name, start_time, end_time, cls_name])

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
    #all
    if args.annotator == "corning":
        split_info_all = get_split_info_corning(args.frame_path, args.annot_file, args.image_fps)
    else:
        split_info_all = get_split_info(args.frame_path, args.annot_file)
    write_split_files(setting_path, split_id, "all", split_info_all)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate split files for DOTA')
    parser.add_argument('--settings', default='./datasets/settings/ucf_crime',
                        help='path to dataset setting files')
    parser.add_argument('--annotator', default='corning',
                        help='annotator of the annotations')
    parser.add_argument('--frame-path', default='base-path/frames',
                        help='path to image files') 
    parser.add_argument('--annot-file', default='base-path/annotations',
                        help='path to annotation files') 
    parser.add_argument('--split-id', default=1, type=int, metavar='S',
                    help='id of the split files, e.g. train{split-id}.txt')  
    parser.add_argument('--image-fps', default=10.0, type=float, metavar='S',
                    help='id of the split files, e.g. train{split-id}.txt') 

    args = parser.parse_args()
    main(args)