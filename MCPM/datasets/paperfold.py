import torch.utils.data as data

import os
import sys
import random
import numpy as np
import cv2


# the slit file are formated as following
# label_name/video_id, start frame, end frame, class_label
def make_dataset(root, source):

    if not os.path.exists(source):
        print("Setting file %s for kinetics100 dataset doesn't exist." % (source))
        sys.exit()
    else:
        clips = []
        labels = []
        with open(source) as split_f:
            data = split_f.readlines()
            for i, line in enumerate(data):
                line_info = line.split(',')
                clip_path = os.path.join(root, line_info[0])
                start_time = int(line_info[1])
                end_time = int(line_info[2])
                if line_info[3].replace("\n","").strip() == "normal":
                    target = 0
                else:
                    target = 1
                item = (clip_path, start_time, end_time, target)
                clips.append(item)
                labels.append(target)
    return clips, labels


def ReadSegmentRGB(path, 
                  start_time, 
                  end_time, 
                  new_height, 
                  new_width, 
                  is_color, 
                  name_pattern,
                  crop=None):
    if is_color:
        cv_read_flag = cv2.IMREAD_COLOR         # > 0
    else:
        cv_read_flag = cv2.IMREAD_GRAYSCALE     # = 0
    interpolation = cv2.INTER_LINEAR

    sampled_list = []

    for frame_index in range(start_time, end_time+1):
        frame_name = name_pattern % (frame_index)
        frame_path = os.path.join(path,frame_name)
        cv_img_origin = cv2.imread(frame_path, cv_read_flag)
        if cv_img_origin is None:
            print("Could not load file %s" % (frame_path))
            sys.exit()
            # TODO: error handling here
        if crop:
            cv_img = cv_img_origin[crop[0]:crop[1], crop[2]:crop[3],:]
        else:
            cv_img = cv_img_origin
        if new_width > 0 and new_height > 0:
            # use OpenCV3, use OpenCV2.4.13 may have error
            cv_img = cv2.resize(cv_img, (new_width, new_height), interpolation)
        
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        sampled_list.append(cv_img)
    clip_input = np.concatenate(sampled_list, axis=2)
    return clip_input


class paperfold(data.Dataset):

    def __init__(self,
                 root,
                 source,
                 modality,
                 name_pattern=None,
                 is_color=True,
                 new_width=0,
                 new_height=0,
                 crop=None,
                 transform=None,
                 target_transform=None,
                 video_transform=None):

        clips, labels = make_dataset(root, source)

        if len(clips) == 0:
            raise(RuntimeError("Found 0 video clips in subfolders of: " + root + "\n"
                               "Check your data directory."))
        assert modality == "rgb", "Only rgb mode is supported now!"
        self.root = root
        self.source = source
        self.modality = modality
        self.clips = clips
        self.labels = labels
        
        # note the default namepattern for the dota dataset is img_%06d.jpg
        if name_pattern:
            self.name_pattern = name_pattern
        else:
            if self.modality == "rgb":
                self.name_pattern = "frame%d.jpg"
            elif self.modality == "flow":
                self.name_pattern = "flow_%s_%06d"

        self.is_color = is_color
        self.new_width = new_width
        self.new_height = new_height
        self.crop = crop

        self.transform = transform
        self.target_transform = target_transform
        self.video_transform = video_transform

    def __getitem__(self, index):
        path, start_time, end_time, target = self.clips[index]

        if self.modality == "rgb":
            clip_input = ReadSegmentRGB(path,
                                        start_time,
                                        end_time,
                                        self.new_height,
                                        self.new_width,
                                        self.is_color,
                                        self.name_pattern,
                                        self.crop
                                        )
            
        else:
            print("No such modality %s" % (self.modality))

        if self.transform is not None:
            clip_input = self.transform(clip_input)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.video_transform is not None:
            clip_input = self.video_transform(clip_input)   
        return clip_input, target
                
    def __len__(self):
        return len(self.clips)