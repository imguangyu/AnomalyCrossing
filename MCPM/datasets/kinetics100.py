import torch.utils.data as data

import os
import sys
import random
import numpy as np
import cv2

CLASS2IDX = {'air drumming': 0,
 'arm wrestling': 1,
 'baking cookies': 2,
 'beatboxing': 3,
 'biking through snow': 4,
 'blasting sand': 5,
 'blowing glass': 6,
 'blowing out candles': 7,
 'bowling': 8,
 'breakdancing': 9,
 'bungee jumping': 10,
 'busking': 11,
 'catching or throwing baseball': 12,
 'cheerleading': 13,
 'cleaning floor': 14,
 'contact juggling': 15,
 'cooking chicken': 16,
 'country line dancing': 17,
 'crossing river': 18,
 'curling hair': 19,
 'cutting watermelon': 20,
 'dancing ballet': 21,
 'dancing charleston': 22,
 'dancing macarena': 23,
 'deadlifting': 24,
 'diving cliff': 25,
 'doing nails': 26,
 'dribbling basketball': 27,
 'driving tractor': 28,
 'drop kicking': 29,
 'dunking basketball': 30,
 'dying hair': 31,
 'eating burger': 32,
 'feeding birds': 33,
 'feeding fish': 34,
 'filling eyebrows': 35,
 'flying kite': 36,
 'folding paper': 37,
 'giving or receiving award': 38,
 'high kick': 39,
 'hopscotch': 40,
 'hula hooping': 41,
 'hurling (sport)': 42,
 'ice skating': 43,
 'javelin throw': 44,
 'jetskiing': 45,
 'jumping into pool': 46,
 'laughing': 47,
 'making snowman': 48,
 'massaging back': 49,
 'mowing lawn': 50,
 'opening bottle': 51,
 'paragliding': 52,
 'playing accordion': 53,
 'playing badminton': 54,
 'playing basketball': 55,
 'playing didgeridoo': 56,
 'playing drums': 57,
 'playing ice hockey': 58,
 'playing keyboard': 59,
 'playing monopoly': 60,
 'playing trombone': 61,
 'playing trumpet': 62,
 'playing ukulele': 63,
 'playing xylophone': 64,
 'presenting weather forecast': 65,
 'punching bag': 66,
 'pushing car': 67,
 'pushing cart': 68,
 'reading book': 69,
 'riding elephant': 70,
 'riding unicycle': 71,
 'scuba diving': 72,
 'shaking head': 73,
 'sharpening pencil': 74,
 'shaving head': 75,
 'shearing sheep': 76,
 'shot put': 77,
 'shuffling cards': 78,
 'side kick': 79,
 'skateboarding': 80,
 'ski jumping': 81,
 'slacklining': 82,
 'sled dog racing': 83,
 'snowboarding': 84,
 'somersaulting': 85,
 'squat': 86,
 'stretching arm': 87,
 'surfing crowd': 88,
 'tap dancing': 89,
 'throwing axe': 90,
 'trapezing': 91,
 'trimming or shaving beard': 92,
 'unboxing': 93,
 'using computer': 94,
 'washing dishes': 95,
 'washing hands': 96,
 'water skiing': 97,
 'waxing legs': 98,
 'weaving basket': 99}

def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    # This line is critical to keep consistency of the class definition
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

# the slit file are formated as following
# label_name/video_id, duration, class_label
def make_dataset(root, source, class_to_idx):

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
                duration = int(line_info[1])
                target = int(line_info[2])
                class_name = os.path.dirname(line_info[0])
                assert class_to_idx[class_name] == target, "lable def concliction at line {} : {}".format(i+1,line)
                item = (clip_path, duration, target)
                clips.append(item)
                labels.append(target)
    return clips, labels


def ReadSegmentRGB(path, offsets, new_height, new_width, new_length, is_color, name_pattern, duration):
    if is_color:
        cv_read_flag = cv2.IMREAD_COLOR         # > 0
    else:
        cv_read_flag = cv2.IMREAD_GRAYSCALE     # = 0
    interpolation = cv2.INTER_LINEAR

    sampled_list = []
    for offset_id in range(len(offsets)):
        offset = offsets[offset_id]
        for length_id in range(1, new_length+1):
            loaded_frame_index = length_id + offset
            moded_loaded_frame_index = loaded_frame_index % (duration + 1)
            if moded_loaded_frame_index == 0:
                moded_loaded_frame_index = (duration + 1)
            frame_name = name_pattern % (moded_loaded_frame_index)
            frame_path = path + "/" + frame_name
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


def ReadSegmentFlow(path, offsets, new_height, new_width, new_length, is_color, name_pattern,duration):
    if is_color:
        cv_read_flag = cv2.IMREAD_COLOR         # > 0
    else:
        cv_read_flag = cv2.IMREAD_GRAYSCALE     # = 0
    interpolation = cv2.INTER_LINEAR

    sampled_list = []
    for offset_id in range(len(offsets)):
        offset = offsets[offset_id]
        for length_id in range(1, new_length+1):
            loaded_frame_index = length_id + offset
            moded_loaded_frame_index = loaded_frame_index % (duration + 1)
            if moded_loaded_frame_index == 0:
                moded_loaded_frame_index = (duration + 1)
            frame_name_x = name_pattern % ("x", moded_loaded_frame_index)
            frame_path_x = path + "/" + frame_name_x
            cv_img_origin_x = cv2.imread(frame_path_x, cv_read_flag)
            frame_name_y = name_pattern % ("y", moded_loaded_frame_index)
            frame_path_y = path + "/" + frame_name_y
            cv_img_origin_y = cv2.imread(frame_path_y, cv_read_flag)
            if cv_img_origin_x is None or cv_img_origin_y is None:
               print("Could not load file %s or %s" % (frame_path_x, frame_path_y))
               sys.exit()
               # TODO: error handling here
            if new_width > 0 and new_height > 0:
                cv_img_x = cv2.resize(cv_img_origin_x, (new_width, new_height), interpolation)
                cv_img_y = cv2.resize(cv_img_origin_y, (new_width, new_height), interpolation)
            else:
                cv_img_x = cv_img_origin_x
                cv_img_y = cv_img_origin_y
            sampled_list.append(np.expand_dims(cv_img_x, 2))
            sampled_list.append(np.expand_dims(cv_img_y, 2))

    clip_input = np.concatenate(sampled_list, axis=2)
    return clip_input


class kinetics100(data.Dataset):

    def __init__(self,
                 root,
                 source,
                 phase,
                 modality,
                 name_pattern=None,
                 is_color=True,
                 num_segments=1,
                 new_length=1,
                 new_width=0,
                 new_height=0,
                 transform=None,
                 target_transform=None,
                 video_transform=None,
                 ensemble_training = False):

        classes, class_to_idx = find_classes(root)
        clips, lablels = make_dataset(root, source, class_to_idx)

        if len(clips) == 0:
            raise(RuntimeError("Found 0 video clips in subfolders of: " + root + "\n"
                               "Check your data directory."))

        self.root = root
        self.source = source
        self.phase = phase
        self.modality = modality

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.clips = clips
        self.labels = lablels
        self.ensemble_training = ensemble_training

        if name_pattern:
            self.name_pattern = name_pattern
        else:
            if self.modality == "rgb":
                self.name_pattern = "img_%05d.jpg"
            elif self.modality == "flow":
                self.name_pattern = "flow_%s_%05d"

        self.is_color = is_color
        self.num_segments = num_segments
        self.new_length = new_length
        self.new_width = new_width
        self.new_height = new_height

        self.transform = transform
        self.target_transform = target_transform
        self.video_transform = video_transform

    def __getitem__(self, index):
        path, duration, target = self.clips[index]
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
                elif duration >= self.new_length:
                    offsets.append(int((seg_id*average_part_length + (seg_id + 1) * average_part_length)/2))
                else:
                    increase = int(duration / self.num_segments)
                    offsets.append(0 + seg_id * increase)
            else:
                print("Only phase train and val are supported.")
        


        if self.modality == "rgb":
            clip_input = ReadSegmentRGB(path,
                                        offsets,
                                        self.new_height,
                                        self.new_width,
                                        self.new_length,
                                        self.is_color,
                                        self.name_pattern,
                                        duration
                                        )
        elif self.modality == "flow":
            clip_input = ReadSegmentFlow(path,
                                        offsets,
                                        self.new_height,
                                        self.new_width,
                                        self.new_length,
                                        self.is_color,
                                        self.name_pattern,
                                        duration
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
