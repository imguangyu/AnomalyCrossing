from PIL import ImageOps, Image
import os
from glob import glob
from torchvision import datasets, models, transforms
from video_transform_pil import *
from datasets import dota_ctx, ucf_crime_ctx
import torch 
import cv2
import models
import torch.nn as nn
from models.baselinefinetune import BaselineFinetune_DA3 as BaselineFinetune_DA
from models.classification_heads import protonet, ClassificationHead, cosineDist
from sklearn import metrics

import time
from PIL import Image
import cv2

import numpy as np

def load_encoder(args):
    
    encoder=models.__dict__[args.arch](modelPath='')
    if args.modelLocation:
        model_path = os.path.join(args.modelLocation,args.saved_model_name) 
        params = torch.load(model_path)
        # print(args.modelLocation)
        encoder.load_state_dict(params['state_dict_encoder'])
    return encoder

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
            transform_list = ['RandomRotation', 'RandomSizedCrop', 'RandomHorizontalFlip', 'ColorJitter', 'ToTensor', 'Normalize']
            #, 'GuassianBlur'
        else:
            transform_list = ['Scale', 'CenterCrop', 'ToTensor', 'Normalize']
        #'ToTensor', 'Normalize' 
        transform_funcs = [ self.parse_transform(x) for x in transform_list]
        transform = transforms.Compose(transform_funcs)
        return transform   

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
                  max_num_nodes=None,
                  mode="start"):
    if is_color:
        cv_read_flag = cv2.IMREAD_COLOR         # > 0
    else:
        cv_read_flag = cv2.IMREAD_GRAYSCALE     # = 0
    interpolation = cv2.INTER_LINEAR

    sampled_ctx = []
    sampled_ctx_id = []
    offset = start_time
    while offset + new_length <= end_time + 1:
        sampled_list = []
        sampled_list_id = []
        for length_id in range(offset, offset + new_length):
            frame_name = name_pattern % (length_id)
            frame_path = os.path.join(path,frame_name)
            # note here the images are loaded using pil and not scaled 
            # and stacked. A list of images will be returned
            img = pil_loader(frame_path)
            sampled_list.append(img)
            sampled_list_id.append(length_id)
        sampled_ctx.append(sampled_list)
        sampled_ctx_id.append(sampled_list_id)
        offset += stride
        if max_num_nodes != None and len(sampled_ctx) >= max_num_nodes:
            break
    return sampled_ctx, sampled_ctx_id

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
    class config:
      pass
    args = config()
    args.frames_path = '/home/liuz2/Py_Work/ActionRecognition/Dataset/UCF_Anomaly/UCF_Crimes/PicFromVid_05_15_2021'
    #'/lstr/home/liuz2/Py_Work/ActionRecognition/Dataset/DOTA_dataset/Detection-of-Traffic-Anomaly-master/Detection-of-Traffic-Anomaly-master/dataset/PicFromVid'
    #'/home/liuz2/Py_Work/ActionRecognition/Dataset/UCF_Anomaly/UCF_Crimes/PicFromVid_05_15_2021'
    args.settings = 'datasets/settings'
    args.split = 2
    args.name_pattern_rgb = '%06d.jpg'
    args.phase = 'all'
    args.dataset = 'ucf_crime'
    #'dota'
    #'ucf_crime'
    args.print_freq = 10

    args.arch = 'rgb_r2plus1d_8f_34_encoder'
    args.modelLocation = './checkpoint/UCF_Crime/DSM/05-17-1536'
    #'./checkpoint/UCF_Crime/DSM/05-23-1222'
    #'./checkpoint/DSM/05-15-2346'
    #'./checkpoint/DSM/BDD/05-16-1746'
    #'./checkpoint/UCF_Crime/DSM/05-17-1536'
    args.saved_model_name = 'ckpt_epoch_160.pth'
    #'ckpt_epoch_200.pth'
    args.device = 'cuda'
    args.save_root = "/home/liuz2/Py_Work/ActionRecognition/Dataset/UCF_Anomaly/UCF_Crimes/CTX_Vectors_8fs4_224_min20_max124_aug_temporal_annoted_corning_DSM_05_17_1536_0160"
    #"/home/liuz2/Py_Work/ActionRecognition/Dataset/UCF_Anomaly/UCF_Crimes/CTX_Vectors_8fs4_224_min20_max124_aug_temporal_annoted_corning_DSM_05_23_1222_0500"
    #'/lstr/home/liuz2/Py_Work/ActionRecognition/Dataset/DOTA_dataset/Detection-of-Traffic-Anomaly-master/Detection-of-Traffic-Anomaly-master/dataset/CTX_Vectors_8fs4_224_min20_aug_DSM_BDD_05_16_1746_0200'
    #"/home/liuz2/Py_Work/ActionRecognition/Dataset/UCF_Anomaly/UCF_Crimes/CTX_Vectors_8fs4_224_min20_max124_aug_temporal_annoted_DSM_05_17_1536_0200"
    #  "/home/liuz2/Py_Work/ActionRecognition/Dataset/UCF_Anomaly/UCF_Crimes/CTX_Vectors_8fs4_224_min20_max124_aug_temporal_annoted_corning"
    #'/lstr/home/liuz2/Py_Work/ActionRecognition/Dataset/DOTA_dataset/Detection-of-Traffic-Anomaly-master/Detection-of-Traffic-Anomaly-master/dataset/CTX_Vectors_8fs4_224_min20_aug_DSM_05_06_1157_0200'
    #"/home/liuz2/Py_Work/ActionRecognition/Dataset/UCF_Anomaly/UCF_Crimes/CTX_Vectors_8fs4_224_min20_max124_aug_temporal_annoted_corning_DSM_05_17_1536_0200"
    
    # print(args.modelLocation)
    #ctx settings
    args.selected_cls = ""
    args.ego_envolve = ""

    stride = args.stride = 4
    width = args.width = 340
    height = args.height = 256
    image_size = args.image_size = 224

    args.min_duration = 20
    args.max_duration = 124
    args.num_augs = 20
    args.aug_folder = 'augmented1'
    args.do_origin_ctx = True
    args.do_aug_ctx = False

    modality=args.arch.split('_')[0]
    if modality == "rgb" or modality == "pose":
        is_color = True
    else:
        is_color = False

    if '64f' in args.arch:
        length = 64
    elif '32f' in args.arch:
        length = 32
    elif '16f' in args.arch:
        length = 16
    elif '8f' in args.arch:
        length = 8
    else:
        length = 16

    print('length %d, input size %d, width %d height %d' %(length, image_size, width, height))
    if args.selected_cls:
        selected_cls = args.selected_cls
    else:
        selected_cls = None
    
    if args.ego_envolve:
        ego_envolve = args.ego_envolve
    else:
        ego_envolve = None 

    #note this is the mean and std for r2plus1d
    clip_mean = [0.43216, 0.394666, 0.37645]
    clip_std = [0.22803, 0.22145, 0.216989] 
    normalize_param = dict(mean= clip_mean , std=clip_std)
    jitter_param = dict(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, consistent=True, p=0.5)
    #seems NEAREST is better than BILINEAR 
    scale_param = dict(size=(width,height), interpolation=Image.NEAREST)
    guassian_blur_param = dict(kernel_size=3, sigma=0.2, p=0.5)
    rand_rotation_param = dict(consistent=True, degree=5, p=0.5, resample = Image.NEAREST)

    setting_file = "%s_%s_split%d.txt" % (args.phase, modality, args.split)
    split_file = os.path.join(args.settings, args.dataset, setting_file)

    device = torch.device(args.device)
    encoder = load_encoder(args)
    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        # print("use data parallel")
        encoder = torch.nn.DataParallel(encoder)
    encoder.to(device)
    encoder.eval()

    avgpool = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))

    trans_loader = TransformLoader(image_size, 
                                normalize_param=normalize_param, 
                                scale_param = scale_param,
                                jitter_param = jitter_param,
                                guassian_blur_param=guassian_blur_param,
                                rand_rotation_param =rand_rotation_param,
                                )
    video_transform = trans_loader.get_composed_transform(aug=False)
    video_transform_aug = trans_loader.get_composed_transform(aug=True)
    
    if args.dataset == "dota":
        clips, labels = dota_ctx.make_dataset(args.frames_path, split_file, ego_envolve, selected_cls, args.min_duration)
    elif args.dataset == "ucf_crime":
        clips, labels = ucf_crime_ctx.make_dataset(args.frames_path, split_file, selected_cls, args.min_duration, args.max_duration)

    print("Num of samples {} selected_cls: {}; ego_envolve: {}".format(len(clips), selected_cls, ego_envolve))
    for index in range(len(clips)):   
        if (index + 1) % args.print_freq == 0:
            print(index+1)
        path, start_time, end_time, target, ego_envolve, selected_cls = clips[index]
        if target == 0:
            label = 'normal'
        else:
            label = 'abnormal'
        folder = os.path.basename(path)
        save_path = os.path.join(args.save_root, folder)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if args.do_aug_ctx:
            aug_path = os.path.join(save_path, args.aug_folder)
            if not os.path.exists(aug_path):
                os.makedirs(aug_path)
    #     duration = end_time - start_time + 1
    #     duration
        sampled_ctx, sampled_ctx_id = ReadSegmentRGB(path, 
                          start_time, 
                          end_time,
                          length, 
                          stride,
                          is_color, 
                          args.name_pattern_rgb,
                          mode="start")
        if args.do_origin_ctx:
            lizx = [video_transform(clip) for clip in sampled_ctx]
            lizx = torch.cat([torch.cat(x, dim=0).view(length,3,image_size,image_size).transpose(1,0).unsqueeze(0) for x in lizx])

            with torch.no_grad():
                ctx_vector = encoder(lizx.to(device))
                ctx_vector = avgpool(ctx_vector).squeeze()
                if ctx_vector.dim() < 2:
                    ctx_vector.unsqueeze(0)
            feature_file = os.path.join(save_path,"ctx_%s.npy" % label)

            np.save(feature_file, ctx_vector.data.cpu().numpy())
        
        if not args.do_aug_ctx:
            continue
        for aug_id in range(args.num_augs):
            # flat the clip list first
            # since we want to apply the same transformer to all the clips
            sampled_ctx_flat = []
            for clip in sampled_ctx:
                for f in clip:
                    sampled_ctx_flat.append(f)
            #every time a new transformation will be applied so the augmented samples will be different
            sampled_ctx_flat = video_transform_aug(sampled_ctx_flat)

            sampled_ctx_aug = []
            for i in range(0,len(sampled_ctx_flat), length):
                sampled_ctx_aug.append(sampled_ctx_flat[i:i+length])
            lizx = [torch.cat(x, dim=0).view(length,3,image_size,image_size).transpose(1,0).unsqueeze(0) for x in sampled_ctx_aug]
            lizx = torch.cat(lizx)
            with torch.no_grad():
                ctx_vector = encoder(lizx.to(device))
                ctx_vector = avgpool(ctx_vector).squeeze()
                if ctx_vector.dim() < 2:
                    ctx_vector.unsqueeze(0)
            feature_file = os.path.join(aug_path,"ctx_%s_%d.npy" % (label, aug_id))

            np.save(feature_file, ctx_vector.data.cpu().numpy())
        # if index > 30:
        #     break