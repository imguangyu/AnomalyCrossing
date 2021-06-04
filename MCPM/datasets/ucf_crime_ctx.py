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

# the slit file are formated as following
#video_id, start frame, end frame, class
def make_dataset(root, source, ego_envolve=None, selected_cls=None, min_duration=None, max_duration=None):

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
                vid_len = end_time - start_time + 1
                #skip too short videos
                if min_duration != None and vid_len < min_duration:
                    continue
                if max_duration != None and vid_len > max_duration:
                    end_time = start_time + max_duration - 1
                if cls_info == "Normal":
                    target = 0
                else:
                    target = 1
                item = (clip_path, start_time, end_time, target, ego_envolve, cls_info)
                # item = (clip_path, start_time, end_time, target, ego_envolve, selected_cls)
                clips.append(item)
                labels.append(target)
    return clips, labels


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
        self.max_duration =  max_duration
        
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