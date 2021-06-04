import torch.utils.data as data

import os
import sys
import random
import numpy as np
import cv2

CLASS2IDX = {'lateral': 0,
 'leave_to_left': 1,
 'leave_to_right': 2,
 'moving_ahead_or_waiting': 3,
 'obstacle': 4,
 'oncoming': 5,
 'pedestrian': 6,
 'start_stop_or_stationary': 7,
 'turning': 8,
 'unknown': 9}

# the slit file are formated as following
# label_name/video_id, duration, class_label
def make_dataset(root, source, ego_envolve=None, selected_cls=None,):

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
                ego_info = line_info[5].replace("\n","").strip()
                cls_info = line_info[4].strip()
                if ego_envolve != None and ego_info.lower() != ego_envolve.lower():
                    continue
                if selected_cls != None and cls_info != selected_cls:
                    continue
                clip_path = os.path.join(root, line_info[0])
                start_time = int(line_info[1])
                end_time = int(line_info[2])
                if line_info[3] == "normal":
                    target = 0
                else:
                    target = 1
                item = (clip_path, start_time, end_time, target, ego_info, cls_info)
                # item = (clip_path, start_time, end_time, target, ego_envolve, selected_cls)
                clips.append(item)
                labels.append(target)
    return clips, labels


def ReadSegmentRGB(path, 
                  start_time, 
                  offsets, 
                  new_height, 
                  new_width, 
                  new_length, 
                  is_color, 
                  name_pattern, 
                  duration,
                  mode="cycle"):
    if is_color:
        cv_read_flag = cv2.IMREAD_COLOR         # > 0
    else:
        cv_read_flag = cv2.IMREAD_GRAYSCALE     # = 0
    interpolation = cv2.INTER_LINEAR

    sampled_list = []
    for offset_id in range(len(offsets)):
        offset = offsets[offset_id]
        for length_id in range(1, new_length+1):
            if mode == "cycle":
                loaded_frame_index = length_id + offset 
                moded_loaded_frame_index = loaded_frame_index % (duration + 1)
                if moded_loaded_frame_index == 0:
                    moded_loaded_frame_index = (duration + 1)
                #frame started from 0 so minus 1
                moded_loaded_frame_index += start_time - 1
            elif mode == "append":
                loaded_frame_index = length_id + offset
                moded_loaded_frame_index = loaded_frame_index if loaded_frame_index <= (duration + 1) else (duration + 1)
                #frame started from 0 so minus 1
                moded_loaded_frame_index += start_time - 1
            else:
                #uniformly sampled offset must be 0
                moded_loaded_frame_index = (length_id-1)*(duration+1) // new_length
                moded_loaded_frame_index += start_time
            frame_name = name_pattern % (moded_loaded_frame_index)
            frame_path = os.path.join(path,frame_name)
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
    clip_input = np.concatenate(sampled_list, axis=2)
    return clip_input


# def ReadSegmentFlow(path, offsets, new_height, new_width, new_length, is_color, name_pattern,duration):
#     if is_color:
#         cv_read_flag = cv2.IMREAD_COLOR         # > 0
#     else:
#         cv_read_flag = cv2.IMREAD_GRAYSCALE     # = 0
#     interpolation = cv2.INTER_LINEAR

#     sampled_list = []
#     for offset_id in range(len(offsets)):
#         offset = offsets[offset_id]
#         for length_id in range(1, new_length+1):
#             loaded_frame_index = length_id + offset
#             moded_loaded_frame_index = loaded_frame_index % (duration + 1)
#             if moded_loaded_frame_index == 0:
#                 moded_loaded_frame_index = (duration + 1)
#             frame_name_x = name_pattern % ("x", moded_loaded_frame_index)
#             frame_path_x = path + "/" + frame_name_x
#             cv_img_origin_x = cv2.imread(frame_path_x, cv_read_flag)
#             frame_name_y = name_pattern % ("y", moded_loaded_frame_index)
#             frame_path_y = path + "/" + frame_name_y
#             cv_img_origin_y = cv2.imread(frame_path_y, cv_read_flag)
#             if cv_img_origin_x is None or cv_img_origin_y is None:
#                print("Could not load file %s or %s" % (frame_path_x, frame_path_y))
#                sys.exit()
#                # TODO: error handling here
#             if new_width > 0 and new_height > 0:
#                 cv_img_x = cv2.resize(cv_img_origin_x, (new_width, new_height), interpolation)
#                 cv_img_y = cv2.resize(cv_img_origin_y, (new_width, new_height), interpolation)
#             else:
#                 cv_img_x = cv_img_origin_x
#                 cv_img_y = cv_img_origin_y
#             sampled_list.append(np.expand_dims(cv_img_x, 2))
#             sampled_list.append(np.expand_dims(cv_img_y, 2))

#     clip_input = np.concatenate(sampled_list, axis=2)
#     return clip_input


class dota(data.Dataset):

    def __init__(self,
                 root,
                 source,
                 phase,
                 modality,
                 ego_envolve=None,
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
                 ensemble_training=False,
                 sample_mode="cycle"):

        clips, lablels = make_dataset(root, source, ego_envolve, selected_cls)

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
        self.sample_mode = sample_mode
        
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
        path, start_time, end_time, target, ego_envolve, selected_cls = self.clips[index]
        duration = end_time - start_time + 1
        duration = duration - 1
        average_duration = int(duration / self.num_segments)
        average_part_length = int(np.floor((duration-self.new_length) / self.num_segments))
        offsets = []
        for seg_id in range(self.num_segments):
            if self.phase == "train":
                if average_duration >= self.new_length:
                    offset = random.randint(0, average_duration - self.new_length)
                    # No +1 because randint(a,b) return a random integer N such that a <= N <= b.
                    offsets.append(offset + seg_id * average_duration)
                elif duration >= self.new_length:
                    offset = random.randint(0, average_part_length)
                    offsets.append(seg_id*average_part_length + offset)
                else:
                    increase = random.randint(0, duration)
                    offsets.append(0 + seg_id * increase)
            elif self.phase == "val":
                if average_duration >= self.new_length:
                    offsets.append(int((average_duration - self.new_length + 1)/2 + seg_id * average_duration))
                # elif duration >= self.new_length:
                #     offsets.append(int((seg_id*average_part_length + (seg_id + 1) * average_part_length)/2))
                else:
                    increase = int(duration / self.num_segments)
                    offsets.append(0 + seg_id * increase)
            else:
                print("Only phase train and val are supported.")
        


        if self.modality == "rgb":
            clip_input = ReadSegmentRGB(path,
                                        start_time,
                                        offsets,
                                        self.new_height,
                                        self.new_width,
                                        self.new_length,
                                        self.is_color,
                                        self.name_pattern,
                                        duration,
                                        self.sample_mode
                                        )
        # elif self.modality == "flow":
        #     clip_input = ReadSegmentFlow(path,
        #                                 start_time,
        #                                 offsets,
        #                                 self.new_height,
        #                                 self.new_width,
        #                                 self.new_length,
        #                                 self.is_color,
        #                                 self.name_pattern,
        #                                 duration
        #                                 )
            
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




class dota2(data.Dataset):

    def __init__(self,
                 root,
                 source,
                 phase,
                 modality,
                 ego_envolve=None,
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
                 ensemble_training=False,
                 sample_mode="cycle"):

        clips, lablels = make_dataset(root, source, ego_envolve, selected_cls)

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
        self.sample_mode = sample_mode
        
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
        path, start_time, end_time, target, ego_envolve, selected_cls = self.clips[index]
        duration = end_time - start_time + 1
        duration = duration - 1
        average_duration = int(duration / self.num_segments)
        average_part_length = int(np.floor((duration-self.new_length) / self.num_segments))
        offsets = []
        for seg_id in range(self.num_segments):
            if self.phase == "train":
                if average_duration >= self.new_length:
                    offset = random.randint(0, average_duration - self.new_length)
                    # No +1 because randint(a,b) return a random integer N such that a <= N <= b.
                    offsets.append(offset + seg_id * average_duration)
                elif duration >= self.new_length:
                    offset = random.randint(0, average_part_length)
                    offsets.append(seg_id*average_part_length + offset)
                else:
                    increase = random.randint(0, duration)
                    offsets.append(0 + seg_id * increase)
            elif self.phase == "val":
                if average_duration >= self.new_length:
                    offsets.append(int((average_duration - self.new_length + 1)/2 + seg_id * average_duration))
                # elif duration >= self.new_length:
                #     offsets.append(int((seg_id*average_part_length + (seg_id + 1) * average_part_length)/2))
                else:
                    increase = int(duration / self.num_segments)
                    offsets.append(0 + seg_id * increase)
            else:
                print("Only phase train and val are supported.")
        


        if self.modality == "rgb":
            clip_input = ReadSegmentRGB(path,
                                        start_time,
                                        offsets,
                                        self.new_height,
                                        self.new_width,
                                        self.new_length,
                                        self.is_color,
                                        self.name_pattern,
                                        duration,
                                        self.sample_mode
                                        )
        # elif self.modality == "flow":
        #     clip_input = ReadSegmentFlow(path,
        #                                 start_time,
        #                                 offsets,
        #                                 self.new_height,
        #                                 self.new_width,
        #                                 self.new_length,
        #                                 self.is_color,
        #                                 self.name_pattern,
        #                                 duration
        #                                 )
            
        else:
            print("No such modality %s" % (self.modality))

        if self.transform is not None:
            clip_input = self.transform(clip_input)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.video_transform is not None:
            clip_input = self.video_transform(clip_input)   
        return clip_input, target, ego_envolve, selected_cls
                


    def __len__(self):
        return len(self.clips)
