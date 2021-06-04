import torch.utils.data as data

import os
import sys
import random
import numpy as np
import cv2

# the slit file are formated as following
#video_id, start frame, end frame, class
def make_dataset(root, source, selected_cls=None,):

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
                cls_info = line_info[3].replace("\n","").strip()
                if selected_cls != None and cls_info != "Normal" and cls_info != selected_cls:
                    continue
                clip_path = os.path.join(root, line_info[0])
                start_time = int(line_info[1])
                end_time = int(line_info[2])
                if cls_info == "Normal":
                    target = 0
                else:
                    target = 1
                item = (clip_path, start_time, end_time, target, cls_info)
                # item = (clip_path, start_time, end_time, target, ego_envolve, selected_cls)
                clips.append(item)
                labels.append(target)
    return clips, labels


def ReadSegmentRGB(path, 
                  start_time, 
                  offset, 
                  duration,
                  new_height, 
                  new_width, 
                  new_length, 
                  is_color, 
                  name_pattern, 
                  ):
    if is_color:
        cv_read_flag = cv2.IMREAD_COLOR         # > 0
    else:
        cv_read_flag = cv2.IMREAD_GRAYSCALE     # = 0
    interpolation = cv2.INTER_LINEAR

    sampled_list = []
    for length_id in range(new_length):
        frame_index = offset + length_id
        frame_index = frame_index % duration
        frame_index += start_time 
        #loop from the start if exceed the end
        frame_name = name_pattern % (frame_index)
        frame_path = os.path.join(path, frame_name)
        cv_img_origin = cv2.imread(frame_path, cv_read_flag)
        if cv_img_origin is None:
           print("Could not load file %s" % (frame_path))
           sys.exit()
           # TODO: error handling here
        if new_width > 0 and new_height > 0:
            # use OpenCV3, use OpenCV2.4.13 may have error
            cv_img = cv2.resize(cv_img_origin, (new_width, new_height), interpolation)
        else:
            cv_img = cv_img_origin
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        sampled_list.append(cv_img)
    #concatenate at the channel dim
    clip_input = np.concatenate(sampled_list, axis=2)
    return clip_input

class ucf_crime(data.Dataset):

    def __init__(self,
                 root,
                 source,
                 phase,
                 modality,
                 ego_envolve=None, #not used actually
                 selected_cls=None,
                 name_pattern=None,
                 is_color=True,
                 num_segments=1,
                 new_length=1,
                 new_width=0,
                 new_height=0,
                 transform=None,
                 target_transform=None,
                 video_transform=None,
                 ensemble_training=False):

        clips, lablels = make_dataset(root, source, selected_cls)

        if len(clips) == 0:
            raise(RuntimeError("Found 0 video clips in subfolders of: " + root + "\n"
                               "Check your data directory."))
        assert modality == "rgb", "Only rgb mode is supported now!"
        assert num_segments == 1, "Only num segments == 1 is supported now!"
        self.root = root
        self.source = source
        self.phase = phase
        self.modality = modality
        self.clips = clips
        self.labels = lablels
        self.ensemble_training = ensemble_training
        
        # note the default namepattern for the dota dataset is img_%06d.jpg
        if name_pattern:
            self.name_pattern = name_pattern
        else:
            if self.modality == "rgb":
                self.name_pattern = "%06d.jpg"
            elif self.modality == "flow":
                self.name_pattern = "flow_%s_%06d"

        self.is_color = is_color
        self.num_segments = num_segments
        self.new_length = new_length
        self.new_width = new_width
        self.new_height = new_height

        self.transform = transform
        self.target_transform = target_transform
        self.video_transform = video_transform

    def __getitem__(self, index):
        path, start_time, end_time, target, selected_cls = self.clips[index]
        duration = end_time - start_time + 1
        offset = 0
        for seg_id in range(self.num_segments):
            if self.phase == "train":
                if duration > self.new_length:
                    offset = random.randint(0, duration - self.new_length)
            elif self.phase == "val":
                if duration > self.new_length:
                    offset = int((duration - self.new_length) / 2)
            else:
                print("Only phase train and val are supported.")
        
        if self.modality == "rgb":
            clip_input = ReadSegmentRGB(path,
                                        start_time,
                                        offset,
                                        duration,
                                        self.new_height,
                                        self.new_width,
                                        self.new_length,
                                        self.is_color,
                                        self.name_pattern,
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
