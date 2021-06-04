import json
import pandas as pd
import glob
import seaborn as sns
import torch
import cv2

from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt

class DotaDataset(Dataset):
    """Dota dataset."""

    def __init__(self, list_name,json_dir,frames_dir,frame_size,antype):
        
        self.frames_dir = frames_dir
        self.frame_size = frame_size

        #read file list
        f=open(list_name)
        file_list = f.read().splitlines()

        #Check if these files exist
        exists = glob.glob(frames_dir+"/*")

        exists = list(map(lambda x: x.replace(frames_dir+"\\",''),exists))

        broken_list = []

        for i,x in enumerate(file_list):
            if x not in exists:
                broken_list.append(x)

        for item in broken_list:
            file_list.remove(item)

        self.jsons = []

        for filename in file_list:
            info = pd.read_json(json_dir+'/'+filename+".json")

            if info['accident_name'][0]==antype:
                self.jsons.append(info)


    def __len__(self):
        return 2*len(self.jsons)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        label = idx%2
        
        info = self.jsons[idx//2]
        file_name = info['video_name'][0]
        anomaly_start = info['anomaly_start'][0]
        anomaly_end = info['anomaly_end'][0]
        anomaly_length = anomaly_end - anomaly_start

        
        #For normal
        if not label:

            if anomaly_start == 0:
                return self.__getitem__(idx+1)
            imgs = []
            
            for i in range(anomaly_start):
                
                img_name = self.frames_dir+"/"+file_name+"/%06d.jpg" % i 
              
                img = cv2.imread(img_name)

                img = cv2.resize(img,(112,112))
               
                imgs.append(img)
            
            xs = []

            

            for i in range(self.frame_size):
               
                xs.append(imgs[i*anomaly_start // self.frame_size])
        #For normal
        else:
            imgs = []
            for i in range(anomaly_start,anomaly_end):
                img_name = self.frames_dir+"/"+file_name+"/%06d.jpg" % i 
                img = cv2.imread(img_name)
                img = cv2.resize(img,(112,112))
                imgs.append(img)
            xs = []
            
            for i in range(self.frame_size):
                xs.append(imgs[i*anomaly_length // self.frame_size])
        
        
        return torch.from_numpy(np.array(xs).swapaxes(0,3).swapaxes(1,3)/255),label