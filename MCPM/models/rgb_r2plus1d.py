"""
Created on Sun Apr 19 23:11:35 2020

@author: esat
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial
from .BERT.bert import BERT, BERT2, BERT3, BERT4, BERT5, BERT6

from .r2plus1d import ( r2plus1d_18,
                        r2plus1d_34_8_ig65m, 
                        r2plus1d_34_32_ig65m, 
                        r2plus1d_34_8_kinetics,
                        r2plus1d_34_32_kinetics, 
                        flow_r2plus1d_34_32_ig65m)

from .representation_flow import resnet_50_rep_flow
from .utils import euclidean_metric

__all__ = ['rgb_r2plus1d_18_encoder',
            'rgb_r2plus1d_8f_34_encoder',
            'rgb_r2plus1d_32f_34_encoder', 
            'rgb_r2plus1d_8f_34_kinetics_encoder',
            'rgb_r2plus1d_32f_34_kinetics_encoder',
            'rgb_r2plus1d_32f_34_meta', 
            'rgb_r2plus1d_32f_34', 
            'rgb_r2plus1d_32f_34_bert10', 
            'rgb_r2plus1d_64f_34_bert10']


class rgb_r2plus1d_18_encoder(nn.Module):
    def __init__(self, modelPath=''):
        super(rgb_r2plus1d_18_encoder, self).__init__()
        self.encoder=nn.Sequential(*list(
            r2plus1d_18( pretrained=True, progress=True).children())[:-2])
        
        for param in self.encoder.parameters():
            param.requires_grad = True

    def forward(self,x):
        x = self.encoder(x)
        return x

class rgb_r2plus1d_8f_34_kinetics_encoder(nn.Module):
    def __init__(self, modelPath=''):
        super(rgb_r2plus1d_8f_34_kinetics_encoder, self).__init__()
        self.encoder=nn.Sequential(*list(
            r2plus1d_34_8_kinetics(400, pretrained=True, progress=True).children())[:-2])
        
        for param in self.encoder.parameters():
            param.requires_grad = True

    def forward(self,x):
        x = self.encoder(x)
        return x

class rgb_r2plus1d_32f_34_kinetics_encoder(nn.Module):
    def __init__(self, modelPath=''):
        super(rgb_r2plus1d_32f_34_kinetics_encoder, self).__init__()
        self.encoder=nn.Sequential(*list(
            r2plus1d_34_32_kinetics(400, pretrained=True, progress=True).children())[:-2])
        
        for param in self.encoder.parameters():
            param.requires_grad = True

    def forward(self,x):
        x = self.encoder(x)
        return x
class rgb_r2plus1d_8f_34_encoder(nn.Module):
    def __init__(self, modelPath=''):
        super(rgb_r2plus1d_8f_34_encoder, self).__init__()
        self.encoder=nn.Sequential(*list(
            r2plus1d_34_8_ig65m(487, pretrained=True, progress=True).children())[:-2])
        
        for param in self.encoder.parameters():
            param.requires_grad = True

    def forward(self,x):
        x = self.encoder(x)
        return x

class rgb_r2plus1d_32f_34_encoder(nn.Module):
    def __init__(self, modelPath=''):
        super(rgb_r2plus1d_32f_34_encoder, self).__init__()
        self.encoder=nn.Sequential(*list(
            r2plus1d_34_32_ig65m(359, pretrained=True, progress=True).children())[:-2])
        
        for param in self.encoder.parameters():
            param.requires_grad = True

    def forward(self,x):
        x = self.encoder(x)
        return x

class rgb_r2plus1d_32f_34_meta(nn.Module):
    def __init__(self, shot , way, temperature, modelPath=''):
        super(rgb_r2plus1d_32f_34_meta, self).__init__()
        self.shot = shot
        self.way = way
        self.temperature = temperature
        self.avgpool = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
        self.encoder=nn.Sequential(*list(
            r2plus1d_34_32_ig65m(359, pretrained=True, progress=True).children())[:-2])
        
        #self.avgpool = nn.AvgPool3d((1, 7, 7), stride=1)

        for param in self.encoder.parameters():
            param.requires_grad = True

    def forward(self, data_shot, data_query):
        proto = self.encoder(data_shot)
        proto = self.avgpool(proto)
        proto = proto.view(self.shot, self.way, -1).mean(dim=0)
        query = self.encoder(data_query)
        query = self.avgpool(query)
        query = query.squeeze()
        logits = euclidean_metric(query, proto) / self.temperature
        return logits



class rgb_r2plus1d_32f_34(nn.Module):
    def __init__(self, num_classes , length, modelPath=''):
        super(rgb_r2plus1d_32f_34, self).__init__()
        self.num_classes=num_classes
        self.dp = nn.Dropout(p=0.8)
        self.avgpool = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
        self.features=nn.Sequential(*list(
            r2plus1d_34_32_ig65m(359, pretrained=True, progress=True).children())[:-2])
        
        #self.avgpool = nn.AvgPool3d((1, 7, 7), stride=1)
        self.fc_action = nn.Linear(512, num_classes)
        for param in self.features.parameters():
            param.requires_grad = True
                
        torch.nn.init.xavier_uniform_(self.fc_action.weight)
        self.fc_action.bias.data.zero_()
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dp(x)
        x = self.fc_action(x)
        return x
    
    def mars_forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x
    
class rgb_r2plus1d_32f_34_bert10(nn.Module):
    def __init__(self, num_classes , length, modelPath=''):
        super(rgb_r2plus1d_32f_34_bert10, self).__init__()
        self.hidden_size=512
        self.n_layers=1
        self.attn_heads=8
        self.num_classes=num_classes
        self.length=length
        self.dp = nn.Dropout(p=0.8)
        
        self.avgpool = nn.AvgPool3d((1, 7, 7), stride=1)
        self.features=nn.Sequential(*list(
            r2plus1d_34_32_ig65m(359, pretrained=True, progress=True).children())[:-2])        
        self.bert = BERT5(self.hidden_size, 4 , hidden=self.hidden_size, n_layers=self.n_layers, attn_heads=self.attn_heads)
        print(sum(p.numel() for p in self.bert.parameters() if p.requires_grad))
        self.fc_action = nn.Linear(self.hidden_size, num_classes)
            
        for param in self.features.parameters():
            param.requires_grad = True

        torch.nn.init.xavier_uniform_(self.fc_action.weight)
        self.fc_action.bias.data.zero_()
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        
        x = x.view(x.size(0), self.hidden_size, 4)
        x = x.transpose(1,2)
        input_vectors=x
        norm = input_vectors.norm(p=2, dim = -1, keepdim=True)
        input_vectors = input_vectors.div(norm)
        output , maskSample = self.bert(x)
        classificationOut = output[:,0,:]
        sequenceOut=output[:,1:,:]
        norm = sequenceOut.norm(p=2, dim = -1, keepdim=True)
        sequenceOut = sequenceOut.div(norm)
        output=self.dp(classificationOut)
        x = self.fc_action(output)
        return x, input_vectors, sequenceOut, maskSample
    
    
class rgb_r2plus1d_64f_34_bert10(nn.Module):
    def __init__(self, num_classes , length, modelPath=''):
        super(rgb_r2plus1d_64f_34_bert10, self).__init__()
        self.hidden_size=512
        self.n_layers=1
        self.attn_heads=8
        self.num_classes=num_classes
        self.length=length
        self.dp = nn.Dropout(p=0.8)
        
        self.avgpool = nn.AvgPool3d((1, 7, 7), stride=1)
        self.features=nn.Sequential(*list(
            r2plus1d_34_32_ig65m(359, pretrained=True, progress=True).children())[:-2])        
        self.bert = BERT5(self.hidden_size, 8 , hidden=self.hidden_size, n_layers=self.n_layers, attn_heads=self.attn_heads)
        print(sum(p.numel() for p in self.bert.parameters() if p.requires_grad))
        self.fc_action = nn.Linear(self.hidden_size, num_classes)
            
        for param in self.features.parameters():
            param.requires_grad = True

        torch.nn.init.xavier_uniform_(self.fc_action.weight)
        self.fc_action.bias.data.zero_()
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        
        x = x.view(x.size(0), self.hidden_size, 8)
        x = x.transpose(1,2)
        input_vectors=x
        norm = input_vectors.norm(p=2, dim = -1, keepdim=True)
        input_vectors = input_vectors.div(norm)
        output , maskSample = self.bert(x)
        classificationOut = output[:,0,:]
        sequenceOut=output[:,1:,:]
        norm = sequenceOut.norm(p=2, dim = -1, keepdim=True)
        sequenceOut = sequenceOut.div(norm)
        output=self.dp(classificationOut)
        x = self.fc_action(output)
        return x, input_vectors, sequenceOut, maskSample
    
    
    
    
