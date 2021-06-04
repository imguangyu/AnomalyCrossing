import torch.utils.data as data
from video_transform_pil import *
from PIL import Image
import os
import sys
import random
import numpy as np
import cv2
from abc import abstractmethod
import torch
from .samplers import CategoriesSampler2 as CategoriesSampler

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
                item = (clip_path, start_time, end_time, target, ego_envolve, selected_cls)
                clips.append(item)
                labels.append(target)
    return clips, labels

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def ReadSegmentRGB(path, 
                  start_time, 
                  offsets, 
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
            # note here the images are loaded using pil and not scaled 
            # and stacked. A list of images will be returned
            img = pil_loader(frame_path)
            sampled_list.append(img)
    return sampled_list

class dota_da(data.Dataset):

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
                                        self.new_length,
                                        self.is_color,
                                        self.name_pattern,
                                        duration,
                                        self.sample_mode
                                        )   
        else:
            print("No such modality %s" % (self.modality))

        if self.transform is not None:
            clip_input = self.transform(clip_input)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.video_transform is not None:
            # note here we need to add scale transformation 
            # since the image clip is not scaled yet
            clip_input = self.video_transform(clip_input)   
        # clip_input = torch.cat(clip_input, axis=2)
        # clip_input = [np.array(i) for i in clip_input]
        return clip_input, target
                
    def __len__(self):
        return len(self.clips)

class dota_da2(data.Dataset):

    def __init__(self,
                 clip_set,
                 n_way,
                 n_shot,
                 n_query,
                 phase,
                 modality,
                 ego_envolve=None,
                 selected_cls=None,
                 name_pattern=None,
                 is_color=True,
                 num_segments=1,
                 new_length=1,
                 transform=None,
                 target_transform=None,
                 video_transform=None,
                 ensemble_training=False,
                 sample_mode="cycle"):

        assert modality == "rgb", "Only rgb mode is supported now!"
        assert num_segments == 1, "Only num segments == 1 is supported now!"
        self.n_way = n_way 
        self.n_shot = n_shot 
        self.n_query = n_query
        self.phase = phase
        self.modality = modality
        self.ensemble_training = ensemble_training
        self.sample_mode = sample_mode
        p = self.n_way * self.n_shot
        if self.phase == "train":
            self.clips = clip_set[:p]
        else:
            self.clips = clip_set[p:]
        
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
            if average_duration >= self.new_length:
                    offsets.append(int((average_duration - self.new_length + 1)/2 + seg_id * average_duration))
                # elif duration >= self.new_length:
                #     offsets.append(int((seg_id*average_part_length + (seg_id + 1) * average_part_length)/2))
            else:
                increase = int(duration / self.num_segments)
                offsets.append(0 + seg_id * increase)


        if self.modality == "rgb":
            clip_input = ReadSegmentRGB(path,
                                        start_time,
                                        offsets,
                                        self.new_length,
                                        self.is_color,
                                        self.name_pattern,
                                        duration,
                                        self.sample_mode
                                        )   
        else:
            print("No such modality %s" % (self.modality))

        if self.transform is not None:
            clip_input = self.transform(clip_input)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.video_transform is not None:
            # note here we need to add scale transformation 
            # since the image clip is not scaled yet
            clip_input = self.video_transform(clip_input)   
        # clip_input = torch.cat(clip_input, axis=2)
        # clip_input = [np.array(i) for i in clip_input]
        return clip_input, target
                


    def __len__(self):
        return len(self.clips)

class TransformLoader:
    def __init__(self, image_size, 
                 normalize_param    = dict(mean= [0.485, 0.456, 0.406] , std=[0.229, 0.224, 0.225]),
                 jitter_param       = dict(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, consistent=False, p=1.0),
                 random_sized_crop_param = dict(consistent=True, p=1.0),
                 center_crop_param = dict(consistent=True),
                 guassian_blur_param = dict(kernel_size=3, sigma=0.2, p=1.0),
                 scale_param = dict(size=(340,256)),
                 rand_hflip_param = dict(consistent=True),
                 rand_rotation_param = dict(consistent=True, degree=15, p=1.0),
                 ):
        self.image_size = image_size
        self.normalize_param = normalize_param
        self.jitter_param = jitter_param
        random_sized_crop_param['size'] = self.image_size
        self.random_sized_crop_param = random_sized_crop_param
        center_crop_param['size'] = self.image_size
        self.center_crop_param = center_crop_param
        self.guassian_blur_param = guassian_blur_param
        self.scale_param = scale_param
        self.rand_hflip_param = rand_hflip_param
        self.rand_rotation_param = rand_rotation_param
    
    def parse_transform(self, transform_type):
        if transform_type=='ColorJitter':
            method = ColorJitter(**self.jitter_param)
        elif transform_type=='RandomSizedCrop':
            method = RandomSizedCrop(**self.random_sized_crop_param)
        elif transform_type=='CenterCrop':
            method = CenterCrop(**self.center_crop_param)
        elif transform_type=='RandomHorizontalFlip':
            method = RandomHorizontalFlip(**self.rand_hflip_param)
        elif transform_type=='GuassianBlur':
            method = GuassianBlur(**self.guassian_blur_param)
        elif transform_type=='Scale':
            method = Scale(**self.scale_param)
        elif transform_type=='RandomRotation':
            method = RandomRotation(**self.rand_rotation_param)
        elif transform_type=='Normalize':
            method = Normalize(**self.normalize_param)
        elif transform_type=='ToTensor':
            method = ToTensor()

        return method

    def get_composed_transform(self, aug = False):
        if aug:
            transform_list = ['RandomRotation', 'RandomSizedCrop', 'RandomHorizontalFlip', 'GuassianBlur', 'ToTensor', 'Normalize']
        else:
            transform_list = ['Scale','CenterCrop', 'ToTensor', 'Normalize']
        #'ToTensor', 'Normalize'
        transform_funcs = [ self.parse_transform(x) for x in transform_list]
        transform = transforms.Compose(transform_funcs)
        return transform

class DataManager(object):
    @abstractmethod
    def get_data_loader(self, data_file, aug):
        pass 

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)

class SetDataManager(DataManager):
    def __init__(self, 
                frames_path,
                split_file,
                image_size, 
                n_way=5, 
                n_support=5, 
                n_query=16, 
                n_eposide=100,
                normalize_param=dict(mean= [0.485, 0.456, 0.406] , 
                                  std=[0.229, 0.224, 0.225]),
                jitter_param=dict(brightness=0.1, 
                                    contrast=0.1, 
                                    saturation=0.1, 
                                    hue=0.1, 
                                    consistent=True,
                                    p=1.0),
                scale_param=dict(size=(340,256)),
                guassian_blur_param=dict(kernel_size=3, sigma=0.2, p=1.0),
                rand_rotation_param=dict(consistent=True, degree=15, p=1.0),
                phase="val",
                modality='rgb',
                ego_envolve=None,
                selected_cls=None,
                name_pattern_rgb='%06d.jpg',
                is_color=True,
                length=32,
                num_seg=1
                ):        
        super(SetDataManager, self).__init__()
        self.image_size = image_size
        # self.new_width = new_width
        # self.new_height = new_height
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = n_query
        self.batch_size = n_support + n_query
        self.n_eposide = n_eposide
        self.normalize_param = normalize_param
        self.jitter_param = jitter_param
        self.scale_param = scale_param
        self.guassian_blur_param = guassian_blur_param
        self.rand_rotation_param = rand_rotation_param

        self.frames_path = frames_path
        self.split_file = split_file
        self.phase = phase
        self.modality = modality
        self.ego_envolve = ego_envolve
        self.selected_cls = selected_cls
        self.name_pattern_rgb = name_pattern_rgb
        self.is_color = is_color
        self.length = length
        self.num_seg = num_seg

    def get_data_loader(self, num_aug = 5, num_workers=8): #parameters that would change on train/val set
        trans_loader = TransformLoader(self.image_size,
                                       scale_param = self.scale_param,
                                       jitter_param = self.jitter_param,
                                       normalize_param = self.normalize_param,
                                       guassian_blur_param=self.guassian_blur_param,
                                       rand_rotation_param=self.rand_rotation_param)
        
        video_transform = trans_loader.get_composed_transform(aug=False)
        dataset = dota_da(root=self.frames_path,
                        source=self.split_file,
                        phase=self.phase,
                        modality=self.modality,
                        ego_envolve=self.ego_envolve,
                        selected_cls=self.selected_cls,
                        name_pattern=self.name_pattern_rgb,
                        is_color=self.is_color,
                        new_length=self.length,
                        video_transform=video_transform,
                        num_segments=self.num_seg)

        sampler = CategoriesSampler(dataset.labels, self.n_eposide, self.n_way, self.batch_size)
        perms = sampler.generate_perm() ##permanent samples

        dataset_list = [dataset] #the first one is without data augmentation
        for i in range(num_aug):
            trans_loader = TransformLoader(self.image_size,
                                       scale_param = self.scale_param,
                                       jitter_param = self.jitter_param,
                                       normalize_param = self.normalize_param,
                                       guassian_blur_param=self.guassian_blur_param,
                                       rand_rotation_param=self.rand_rotation_param)
            video_transform = trans_loader.get_composed_transform(aug=True)
            dataset = dota_da(root=self.frames_path,
                        source=self.split_file,
                        phase=self.phase,
                        modality=self.modality,
                        ego_envolve=self.ego_envolve,
                        selected_cls=self.selected_cls,
                        name_pattern=self.name_pattern_rgb,
                        is_color=self.is_color,
                        new_length=self.length,
                        video_transform=video_transform,
                        num_segments=self.num_seg)
            dataset_list.append(dataset)
        dataset_chain = ConcatDataset(dataset_list)
        data_loader_params = dict(batch_sampler = sampler, shuffle = False, num_workers = num_workers, pin_memory = True)
        data_loader = torch.utils.data.DataLoader(dataset_chain, **data_loader_params)

        return data_loader


class TransformLoader2:
    def __init__(self, image_size, 
                 normalize_param    = dict(mean= [0.485, 0.456, 0.406] , std=[0.229, 0.224, 0.225]),
                 jitter_param       = dict(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, consistent=False, p=1.0),
                 random_sized_crop_param = dict(consistent=True, p=1.0),
                 center_crop_param = dict(consistent=True),
                 guassian_blur_param = dict(kernel_size=3, sigma=0.2, p=1.0),
                 scale_param = dict(size=(340,256)),
                 rand_hflip_param = dict(consistent=True),
                 rand_rotation_param = dict(consistent=True, degree=15, p=1.0),
                 ):
        self.image_size = image_size
        self.normalize_param = normalize_param
        self.jitter_param = jitter_param
        random_sized_crop_param['size'] = self.image_size
        self.random_sized_crop_param = random_sized_crop_param
        center_crop_param['size'] = self.image_size
        self.center_crop_param = center_crop_param
        self.guassian_blur_param = guassian_blur_param
        self.scale_param = scale_param
        self.rand_hflip_param = rand_hflip_param
        self.rand_rotation_param = rand_rotation_param
    
    def parse_transform(self, transform_type):
        if transform_type=='ColorJitter':
            method = ColorJitter(**self.jitter_param)
        elif transform_type=='RandomSizedCrop':
            method = RandomSizedCrop(**self.random_sized_crop_param)
        elif transform_type=='CenterCrop':
            method = CenterCrop(**self.center_crop_param)
        elif transform_type=='RandomHorizontalFlip':
            method = RandomHorizontalFlip(**self.rand_hflip_param)
        elif transform_type=='GuassianBlur':
            method = GuassianBlur(**self.guassian_blur_param)
        elif transform_type=='Scale':
            method = Scale(**self.scale_param)
        elif transform_type=='RandomRotation':
            method = RandomRotation(**self.rand_rotation_param)
        elif transform_type=='Normalize':
            method = Normalize(**self.normalize_param)
        elif transform_type=='ToTensor':
            method = ToTensor()

        return method

    def get_composed_transform(self, aug = False):
        if aug:
            # if use RandomRotation then should put it as the first one since it may change the image size
            transform_list = ['RandomRotation', 'RandomSizedCrop', 'RandomHorizontalFlip', 'GuassianBlur', 'ToTensor', 'Normalize']
            # 'ColorJitter',
        else:
            transform_list = ['Scale','CenterCrop', 'ToTensor', 'Normalize']
            #
        #'ToTensor', 'Normalize'
        transform_funcs = [ self.parse_transform(x) for x in transform_list]
        transform = transforms.Compose(transform_funcs)
        return transform

class TransformLoader3:
    def __init__(self, image_size, 
                 normalize_param    = dict(mean= [0.485, 0.456, 0.406] , std=[0.229, 0.224, 0.225]),
                 jitter_param       = dict(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, consistent=False, p=1.0),
                 random_sized_crop_param = dict(consistent=True, p=1.0),
                 center_crop_param = dict(consistent=True),
                 guassian_blur_param = dict(kernel_size=3, sigma=0.2, p=1.0),
                 scale_param = dict(size=(340,256)),
                 rand_hflip_param = dict(consistent=True),
                 rand_rotation_param = dict(consistent=True, degree=15, p=1.0),
                 ):
        self.image_size = image_size
        self.normalize_param = normalize_param
        self.jitter_param = jitter_param
        random_sized_crop_param['size'] = self.image_size
        self.random_sized_crop_param = random_sized_crop_param
        center_crop_param['size'] = self.image_size
        self.center_crop_param = center_crop_param
        self.guassian_blur_param = guassian_blur_param
        self.scale_param = scale_param
        self.rand_hflip_param = rand_hflip_param
        self.rand_rotation_param = rand_rotation_param
    
    def parse_transform(self, transform_type):
        if transform_type=='ColorJitter':
            method = ColorJitter(**self.jitter_param)
        elif transform_type=='RandomSizedCrop':
            method = RandomSizedCrop(**self.random_sized_crop_param)
        elif transform_type=='CenterCrop':
            method = CenterCrop(**self.center_crop_param)
        elif transform_type=='RandomHorizontalFlip':
            method = RandomHorizontalFlip(**self.rand_hflip_param)
        elif transform_type=='GuassianBlur':
            method = GuassianBlur(**self.guassian_blur_param)
        elif transform_type=='Scale':
            method = Scale(**self.scale_param)
        elif transform_type=='RandomRotation':
            method = RandomRotation(**self.rand_rotation_param)
        elif transform_type=='Normalize':
            method = Normalize(**self.normalize_param)
        elif transform_type=='ToTensor':
            method = ToTensor()

        return method

    def get_composed_transform(self, aug = False):
        if aug:
            # if use RandomRotation then should put it as the first one since it may change the image size
            transform_list = ['RandomRotation', 'RandomSizedCrop', 'RandomHorizontalFlip', 'GuassianBlur', 'ToTensor', 'Normalize']
            # 'ColorJitter',
        else:
            transform_list = ['Scale','CenterCrop', 'ToTensor', 'Normalize']
            #
        #,'ToTensor', 'Normalize'
        transform_funcs = [ self.parse_transform(x) for x in transform_list]
        transform = transforms.Compose(transform_funcs)
        return transform

class SetDataManager2(DataManager):
    def __init__(self, 
                frames_path,
                split_file,
                image_size, 
                n_way=5, 
                n_support=5, 
                n_query=16, 
                n_eposide=100,
                normalize_param=dict(mean= [0.485, 0.456, 0.406] , 
                                  std=[0.229, 0.224, 0.225]),
                jitter_param=dict(brightness=0.1, 
                                    contrast=0.1, 
                                    saturation=0.1, 
                                    hue=0.1, 
                                    consistent=False,
                                    p=1.0),
                scale_param=dict(size=(340,256)),
                guassian_blur_param=dict(kernel_size=3, sigma=0.2, p=1.0),
                rand_rotation_param=dict(consistent=True, degree=15, p=1.0),
                phase="val",
                modality='rgb',
                ego_envolve=None,
                selected_cls=None,
                name_pattern_rgb='%06d.jpg',
                is_color=True,
                length=32,
                num_seg=1
                ):        
        super(SetDataManager2, self).__init__()
        self.image_size = image_size
        # self.new_width = new_width
        # self.new_height = new_height
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = n_query
        self.batch_size = n_support + n_query
        self.n_eposide = n_eposide
        self.normalize_param = normalize_param
        self.jitter_param = jitter_param
        self.scale_param = scale_param
        self.guassian_blur_param = guassian_blur_param
        self.rand_rotation_param = rand_rotation_param

        self.frames_path = frames_path
        self.split_file = split_file
        self.phase = phase
        self.modality = modality
        self.ego_envolve = ego_envolve
        self.selected_cls = selected_cls
        self.name_pattern_rgb = name_pattern_rgb
        self.is_color = is_color
        self.length = length
        self.num_seg = num_seg

    def get_data_loader(self, num_aug = 5, num_workers=8): #parameters that would change on train/val set
        trans_loader = TransformLoader2(self.image_size,
                                       scale_param = self.scale_param,
                                       jitter_param = self.jitter_param,
                                       normalize_param = self.normalize_param,
                                       guassian_blur_param=self.guassian_blur_param,
                                       rand_rotation_param=self.rand_rotation_param)
        
        video_transform = trans_loader.get_composed_transform(aug=False)
        dataset = dota_da(root=self.frames_path,
                        source=self.split_file,
                        phase=self.phase,
                        modality=self.modality,
                        ego_envolve=self.ego_envolve,
                        selected_cls=self.selected_cls,
                        name_pattern=self.name_pattern_rgb,
                        is_color=self.is_color,
                        new_length=self.length,
                        video_transform=video_transform,
                        num_segments=self.num_seg)

        sampler = CategoriesSampler(dataset.labels, self.n_eposide, self.n_way, self.batch_size)
        perms = sampler.generate_perm() ##permanent samples

        dataset_list = [dataset] #the first one is without data augmentation
        for i in range(num_aug):
            trans_loader = TransformLoader2(self.image_size,
                                       scale_param = self.scale_param,
                                       jitter_param = self.jitter_param,
                                       normalize_param = self.normalize_param,
                                       guassian_blur_param=self.guassian_blur_param,
                                       rand_rotation_param=self.rand_rotation_param)
            video_transform = trans_loader.get_composed_transform(aug=True)
            dataset = dota_da(root=self.frames_path,
                        source=self.split_file,
                        phase=self.phase,
                        modality=self.modality,
                        ego_envolve=self.ego_envolve,
                        selected_cls=self.selected_cls,
                        name_pattern=self.name_pattern_rgb,
                        is_color=self.is_color,
                        new_length=self.length,
                        video_transform=video_transform,
                        num_segments=self.num_seg)
            dataset_list.append(dataset)
        dataset_chain = ConcatDataset(dataset_list)
        data_loader_params = dict(batch_sampler = sampler, shuffle = False, num_workers = num_workers, pin_memory = True)
        data_loader = torch.utils.data.DataLoader(dataset_chain, **data_loader_params)

        return data_loader

class WorksetDA:
    def __init__(self, root, source, n_batch, n_cls, n_per, ego_envolve=None, selected_cls=None):
        self.root = root
        self.source = source
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per
        self.ego_envolve = ego_envolve
        self.selected_cls = selected_cls
        
        self.clips, labels = self.make_dataset(self.root, self.source, self.ego_envolve, self.selected_cls)
        self.label_set = set(labels)
        self.labels = np.array(labels)
        
        self.m_ind = []
        for i in self.label_set:
            ind = np.argwhere(self.labels == i).reshape(-1)
            self.m_ind.append(ind)
        
        self.generate_perm()
        
    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
        for i in range(self.n_batch):
            yield [self.clips[idx] for  idx in self.perms[i]], [self.labels[idx] for idx in self.perms[i]]
            
    def __getitem__(self, indx):
        return [self.clips[i] for i in self.perms[indx]], [self.labels[i] for i in self.perms[indx]]
    
    def generate_perm(self):
        self.perms = []
        for i_batch in range(self.n_batch):
            batch = []
            classes = np.random.permutation(len(self.m_ind))[:self.n_cls]
            for c in classes:
                l = self.m_ind[c]
                pos = np.random.permutation(len(l))[:self.n_per]
                batch.append(l[pos])
            # Note this transpose is very important 
            # it determines how we should generate labels in training 
            # before transpose the batch matrix is w by (s+q)
            # after transpose the dim is (s+q) + w 
            # so the flatten is go through each classes first 
            # so the labels should be range(w) * repeat(q)
            self.perms.append(np.stack(batch).transpose().reshape(-1))
            
    def make_dataset(self, root, source, ego_envolve=None, selected_cls=None,):

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
                    item = (clip_path, start_time, end_time, target, ego_envolve, selected_cls)
                    clips.append(item)
                    labels.append(target)
        return clips, labels