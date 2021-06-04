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
def make_dataset(root, 
                source,
                ego_envolve=None, 
                selected_cls=None,
                min_duration=None,
                max_duration=None):

    if not os.path.exists(source):
        print("Setting file %s for dota dataset doesn't exist." % (source))
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
                #skip too short videos
                if min_duration != None and end_time - start_time + 1 < min_duration:
                    continue
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
                  end_time,
                  new_length, 
                  stride,
                  is_color, 
                  name_pattern, 
                  mode="start"):
    if is_color:
        cv_read_flag = cv2.IMREAD_COLOR         # > 0
    else:
        cv_read_flag = cv2.IMREAD_GRAYSCALE     # = 0
    interpolation = cv2.INTER_LINEAR

    sampled_ctx = []
    offset = start_time
    while offset + new_length <= end_time + 1:
        sampled_list = []
        for length_id in range(offset, offset + new_length):
            frame_name = name_pattern % (length_id)
            frame_path = os.path.join(path,frame_name)
            # note here the images are loaded using pil and not scaled 
            # and stacked. A list of images will be returned
            img = pil_loader(frame_path)
            sampled_list.append(img)
        sampled_ctx.append(sampled_list)
        offset += stride
    return sampled_ctx

def ReadCTXVec(path, target, name_pattern, file_format='npy'):
    if file_format=='npy':
        if target == 0:
            label = "normal"
        else:
            label = "abnormal"
        file = os.path.join(path,name_pattern % label)
        with open(file, 'rb') as f:
            ctx_vector = np.load(f)
    else:
        print("File format for %s is not supported yet." % (path))
        sys.exit()
    return ctx_vector

def ReadCTXVec_Aug(path, target, aug_id, name_pattern, file_format='npy'):
    if file_format=='npy':
        if target == 0:
            label = "normal"
        else:
            label = "abnormal"
        file = os.path.join(path,name_pattern % (label, aug_id))
        with open(file, 'rb') as f:
            ctx_vector = np.load(f)
    else:
        print("File format for %s is not supported yet." % (path))
        sys.exit()
    return ctx_vector

class data_ctx(data.Dataset):

    def __init__(self,
                 root,
                 source,
                 ego_envolve=None,
                 selected_cls=None,
                 name_pattern=None,
                 min_duration=None,
                 max_duration=None,
                 file_format='npy'):

        clips, lablels = make_dataset(root, source, ego_envolve, selected_cls,min_duration)

        if len(clips) == 0:
            raise(RuntimeError("Found 0 video clips in subfolders of: " + root + "\n"
                               "Check your data directory."))
        self.root = root
        self.source = source
        self.clips = clips
        self.labels = lablels
        self.file_format = file_format
        self.min_duration = min_duration
        self.max_duration = max_duration

        # note the default namepattern for the dota dataset is img_%06d.jpg
        if name_pattern:
            self.name_pattern = name_pattern
        else:
            self.name_pattern = "ctx_%s.npy"

    def __getitem__(self, index):
        path, start_time, end_time, target, ego_envolve, selected_cls = self.clips[index]
        duration = end_time - start_time + 1
      
        ctx_vector = ReadCTXVec(path, target, self.name_pattern, self.file_format)

        return ctx_vector, target
                
    def __len__(self):
        return len(self.clips)

class ctx_set(data.Dataset):

    def __init__(self,
                 clip_set,
                 n_way,
                 n_shot,
                 n_query,
                 phase,
                 num_aug=0,
                 aug_folder=None,
                 name_pattern_aug=None,
                 ego_envolve=None,
                 selected_cls=None,
                 name_pattern=None,
                 file_format='npy'):

        self.n_way = n_way 
        self.n_shot = n_shot 
        self.n_query = n_query
        self.phase = phase

        self.file_format = file_format
        p = self.n_way * self.n_shot
        if self.phase == "train":
            self.clips = clip_set[:p]
        elif self.phase == "eval":
            self.clips = clip_set[p:]
        elif self.phase == "all":
            self.clips = clip_set
        else:
            print("Unknown phase!")
        
        
        # note the default namepattern for the dota dataset is img_%06d.jpg
        if name_pattern:
            self.name_pattern = name_pattern
        else:
            self.name_pattern = "ctx_%s.npy"
        self.num_aug = num_aug
        self.aug_folder = aug_folder

        if name_pattern_aug:
            self.name_pattern_aug = name_pattern_aug
        else:
            self.name_pattern_aug = "ctx_%s_%d.npy"

    def __getitem__(self, index):
        path, start_time, end_time, target, ego_envolve, selected_cls = self.clips[index]
        duration = end_time - start_time + 1
      
        ctx_vector = ReadCTXVec(path, target, self.name_pattern, self.file_format)
        ctx_vector_aug = []
        if self.aug_folder != None:
            aug_path=os.path.join(path, self.aug_folder)
        for i in range(self.num_aug):
            ctx_vector_aug.append(ReadCTXVec_Aug(aug_path, target, i, self.name_pattern_aug, self.file_format))

        return ctx_vector, target, ctx_vector_aug
                
    def __len__(self):
        return len(self.clips)

class TransformLoader:
    def __init__(self, image_size, 
                 normalize_param    = dict(mean= [0.485, 0.456, 0.406] , std=[0.229, 0.224, 0.225]),
                 jitter_param       = dict(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, consistent=False, p=1.0),
                 random_sized_crop_param = dict(consistent=True, p=1.0),
                 center_crop_param = dict(consistent=True),
                 guassian_blur_param = dict(kernel_size=3, sigma=0.2, p=1.0),
                 scale_param = dict(size=(340,256), ),
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


class Workset:
    def __init__(self, 
                root, 
                source, 
                n_batch, 
                n_cls, 
                n_shot, 
                n_query,
                ego_envolve=None, 
                selected_cls=None, 
                min_duration=None,
                max_duration=None):

        self.root = root
        self.source = source
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_shot + n_query
        self.n_shot = n_shot
        self.n_query = n_query
        self.ego_envolve = ego_envolve
        self.selected_cls = selected_cls
        self.min_duration = min_duration
        self.max_duration = max_duration
        
        self.clips, labels = make_dataset(self.root, 
                                        self.source, 
                                        self.ego_envolve, 
                                        self.selected_cls,
                                        self.min_duration,
                                        self.max_duration)
        
        self.label_set = set(labels)
        self.labels = np.array(labels)
        # print(len(self.clips), self.label_set)

        self.m_ind = []
        for i in self.label_set:
            ind = np.argwhere(self.labels == i).reshape(-1)
            assert len(ind) > self.n_shot, "Only have {} samples. Not enough for {} shot {} query".format(len(ind), self.n_shot, self.n_query)
            self.m_ind.append(ind)
        
        self.generate_perm2()
        
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
    
    def generate_perm2(self):
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
            try:
                out = np.stack(batch).transpose().reshape(-1)
            except:
                support = [_[:self.n_shot] for _ in batch]
                query = [_[self.n_shot:] for _ in batch]
                support = np.stack(support).transpose().reshape(-1)
                query = np.concatenate(query)
                out = np.concatenate((support, query))
                #note now for the query part the labels are not range(w) * repeat(q)
            self.perms.append(out)
            
    # def make_dataset(self, 
    #                  root, 
    #                  source, 
    #                  ego_envolve=None, 
    #                  selected_cls=None,
    #                  min_duration=None):

    #     if not os.path.exists(source):
    #         print("Setting file %s for kinetics100 dataset doesn't exist." % (source))
    #         sys.exit()
    #     else:
    #         clips = []
    #         labels = []
    #         with open(source) as split_f:
    #             data = split_f.readlines()
    #             for i, line in enumerate(data):
    #                 line_info = line.split(',')
    #                 ego_info = line_info[5].replace("\n","").strip()
    #                 cls_info = line_info[4].strip()
    #                 if ego_envolve != None and ego_info.lower() != ego_envolve.lower():
    #                     continue
    #                 if selected_cls != None and cls_info != selected_cls:
    #                     continue
    #                 clip_path = os.path.join(root, line_info[0])
    #                 start_time = int(line_info[1])
    #                 end_time = int(line_info[2])
    #                 #skip too short videos
    #                 if min_duration != None and end_time - start_time + 1 < min_duration:
    #                     continue
    #                 if line_info[3] == "normal":
    #                     target = 0
    #                 else:
    #                     target = 1
    #                 item = (clip_path, start_time, end_time, target, ego_envolve, selected_cls)
    #                 clips.append(item)
    #                 labels.append(target)
    #     return clips, labels