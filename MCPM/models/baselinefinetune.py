import numpy as np
import torch 
import torch.nn as nn
from .classification_heads import protonet, ClassificationHead, cosineDist
from sklearn.neighbors import NearestNeighbors
from scipy.linalg import sqrtm
import models
import os
class BaselineFinetune():
    def __init__(self, 
        device,
        n_way, 
        n_support, 
        cls_head_type, 
        n_query=None, 
        batch_size=4,
        num_epochs=301,
        threshold=0.5,
        temperature=128,
        fc_inter_dim=128
        ):
        self.device = device
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = n_query
        self.cls_head_type = cls_head_type
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.threshold = threshold 
        self.temperature = temperature
        self.fc_inter_dim = fc_inter_dim

    def set_forward_adaptation_lp(self,x,is_feature=True,k_lp=10,delta=0.1,alpha=0.5):
        assert is_feature == True, 'Baseline only support testing with feature'
        assert self.cls_head_type in ['fc', 'cosine_dist'], "forward adaptation only support fc and cosine distance classfication head!"
        z_support, z_query, y_support, y_query  = self.parse_feature(x,is_feature)
        self.feat_dim = z_support.size()[1]

        if self.cls_head_type == 'fc':
            self.cls_head= nn.Linear(self.feat_dim, self.n_way)
        elif self.cls_head_type == 'cosine_dist':        
            self.cls_head = cosineDist(self.feat_dim, self.n_way)

        self.cls_head.to(self.device)

        set_optimizer = torch.optim.SGD(self.cls_head.parameters(), lr = 0.01, momentum=0.9, dampening=0.9, weight_decay=0.001)

        loss_function = nn.CrossEntropyLoss()
        # loss_function = loss_function.cuda()
        
        batch_size = self.batch_size
        support_size = self.n_way* self.n_support
        self.cls_head.train()

        for epoch in range(self.num_epochs):
            rand_id = np.random.permutation(support_size)
            for i in range(0, support_size, batch_size):
                set_optimizer.zero_grad()
                selected_id = torch.from_numpy( rand_id[i: min(i+batch_size, support_size) ]).to(self.device)
                z_batch = z_support[selected_id]
                y_batch = y_support[selected_id] 
                scores = self.cls_head(z_batch)
                loss = loss_function(scores,y_batch)
                loss.backward()
                set_optimizer.step()
        
        th = self.threshold 
        self.cls_head.eval()
        with torch.no_grad():
          scores_support = self.cls_head(z_support)
          pred_tr = torch.argmax((torch.softmax(scores_support,-1)>th).float(), -1).cpu().numpy()
          train_acc = np.mean(pred_tr == y_support.cpu().numpy())
          
          scores_query = self.cls_head(z_query)
        
        x_lp = z_query.cpu().numpy()
        y_lp = torch.softmax(scores_query, 1).cpu().numpy()
        ynew_lp = self.label_prop(x_lp, y_lp, self.n_way, k_lp, delta, alpha)

        pred_eval = np.argmax((ynew_lp>th), -1)
        eval_acc = np.mean(pred_eval == y_query.cpu().numpy())

        return eval_acc, train_acc, pred_eval

    def set_forward_lp(self,x,is_feature=True,k_lp=10,delta=0.1,alpha=0.5):
        assert is_feature == True, 'Baseline only support testing with feature'
        assert self.cls_head_type not in ['fc', 'cosine_dist'], "fc and cosine distance classfication head need to be adapted!"
        z_support, z_query, y_support, y_query  = self.parse_feature(x,is_feature)
        if self.cls_head_type == "protonet":
          self.cls_head = protonet(self.n_support,  self.n_way, self.temperature)
          self.cls_head.to(self.device)
          self.cls_head.eval()
          with torch.no_grad():
            scores_query, scores_support = self.cls_head.set_forward_feature(z_support, z_query)
        else:
          self.cls_head = ClassificationHead(base_learner=self.cls_head_type, 
                                             n_shot=self.n_support, 
                                             n_way=self.n_way, 
                                             enable_scale=False)
          self.cls_head.to(self.device)
          self.cls_head.eval()
          with torch.no_grad():
            scores_query, scores_support = self.cls_head.set_forward_feature(z_support, z_query, y_support)
        
        
        x_lp = z_query.cpu().numpy()
        y_lp = torch.softmax(scores_query, 1).cpu().numpy()
        ynew_lp = self.label_prop(x_lp, y_lp, self.n_way, k_lp, delta, alpha)

        th = self.threshold
        pred_tr = torch.argmax((torch.softmax(scores_support,-1)>th).float(), -1).cpu().numpy()
        train_acc = np.mean(pred_tr == y_support.cpu().numpy())
        
        pred_eval = np.argmax((ynew_lp>th), -1)
        eval_acc = np.mean(pred_eval == y_query.cpu().numpy())

        return eval_acc, train_acc, pred_eval 

    def label_prop(self, x_lp, y_lp, n_way, k_lp=10, delta=0.2, alpha=0.5):
        neigh = NearestNeighbors(n_neighbors=k_lp)
        neigh.fit(x_lp)
        d_lp, idx_lp = neigh.kneighbors(x_lp)
        d_lp = np.power(d_lp, 2)
        sigma2_lp = np.mean(d_lp)
        n_lp = len(y_lp)
        del_n = int(n_lp * (1.0 - delta))
        for i in range(n_way):
            yi = y_lp[:, i]
            top_del_idx = np.argsort(yi)[0:del_n]
            y_lp[top_del_idx, i] = 0
            
        w_lp = np.zeros((n_lp, n_lp))
        for i in range(n_lp):
            for j in range(k_lp):
                xj = idx_lp[i, j]
                w_lp[i, xj] = np.exp(-d_lp[i, j] / (2 * sigma2_lp))
                w_lp[xj, i] = np.exp(-d_lp[i, j] / (2 * sigma2_lp))
        q_lp = np.diag(np.sum(w_lp, axis=1))
        q2_lp = sqrtm(q_lp)
        q2_lp = np.linalg.inv(q2_lp)
        L_lp = np.matmul(np.matmul(q2_lp, w_lp), q2_lp)
        a_lp = np.eye(n_lp) - alpha * L_lp
        a_lp = np.linalg.inv(a_lp)
        # return the new predictions
        return np.matmul(a_lp, y_lp)

    def set_forward(self,x,is_feature = True):
        assert is_feature == True, 'Baseline only support testing with feature'
        assert self.cls_head_type not in ['fc', 'fc2', 'cosine_dist'], "fc and cosine distance classfication head need to be adapted!"
        z_support, z_query, y_support, y_query  = self.parse_feature(x,is_feature)
        if self.cls_head_type == "protonet":
          self.cls_head = protonet(self.n_support,  self.n_way, self.temperature)
          self.cls_head.to(self.device)
          self.cls_head.eval()
          with torch.no_grad():
            scores_query, scores_support = self.cls_head.set_forward_feature(z_support, z_query)
        else:
          self.cls_head = ClassificationHead(base_learner=self.cls_head_type, 
                                             n_shot=self.n_support, 
                                             n_way=self.n_way, 
                                             enable_scale=False)
          self.cls_head.to(self.device)
          self.cls_head.eval()
          with torch.no_grad():
            scores_query, scores_support = self.cls_head.set_forward_feature(z_support, z_query, y_support)
        
        th = self.threshold
        pred_tr = torch.argmax((torch.softmax(scores_support,-1)>th).float(), -1).cpu().numpy()
        train_acc = np.mean(pred_tr == y_support.cpu().numpy())
        
        pred_eval = torch.argmax((torch.softmax(scores_query,-1)>th).float(), -1).cpu().numpy()
        eval_acc = np.mean(pred_eval == y_query.cpu().numpy())

        return eval_acc, train_acc, pred_eval 
    
    def set_forward_video(self, support_dataset, query_dataset, mode='mean'):
        assert self.cls_head_type not in ['fc', 'fc2', 'cosine_dist'], "fc and cosine distance classfication head need to be adapted!"
        support_size = len(support_dataset)
        inputs_support = []
        target_support = []
        #inputs dim [np, nf]
        for inputs, target, _ in support_dataset:
            inputs = torch.tensor(inputs)
            index = inputs.size()[0] // 2
            inputs_support.append(inputs[index].unsqueeze(0))
            target_support.append(target)
    
        self.feat_dim = inputs.size()[1]
        cls_label = sorted(set(target_support))
        support_set = []
        support_label = []
        for c in cls_label:
            for i in range(support_size):
                if target_support[i] == c:
                    support_set.append(inputs_support[i])
                    support_label.append(target_support[i])

        support_set = torch.cat(support_set).to(self.device)
        y_support = torch.tensor(support_label).to(self.device)
        if self.cls_head_type == "protonet":
          self.cls_head = protonet(self.n_support,  self.n_way, self.temperature)
        else:
          self.cls_head = ClassificationHead(base_learner=self.cls_head_type, 
                                             n_shot=self.n_support, 
                                             n_way=self.n_way, 
                                             enable_scale=False)
        self.cls_head.to(self.device)
        self.cls_head.eval()
        scores_query = []
        target_query = []
        with torch.no_grad():
            for i, (inputs, target, _) in enumerate(query_dataset): 
                inputs = torch.tensor(inputs).to(self.device)
                scores, scores_support = self.cls_head.set_forward_feature(support_set, inputs, y_support)
                if mode == 'mean':
                    scores = torch.mean(scores, dim=0).unsqueeze(0)
                else:
                    scores = torch.max(scores, dim=0)[0].unsqueeze(0)
                scores_query.append(scores)
                target_query.append(target)

        th = self.threshold 
        scores_query = torch.cat(scores_query) 
        pred_eval = torch.argmax((torch.softmax(scores_query,-1)>th).float(), -1).cpu().numpy()
        pred_tr = torch.argmax((torch.softmax(scores_support,-1)>th).float(), -1).cpu().numpy()     
        eval_acc = np.mean(pred_eval == target_query)
        train_acc = np.mean(pred_tr == support_label)
         
        return eval_acc, train_acc, pred_eval, target_query

    def set_forward_adaptation(self,x,is_feature = True):
        assert is_feature == True, 'Baseline only support testing with feature'
        assert self.cls_head_type in ['fc', 'fc2', 'cosine_dist'], "forward adaptation only support fc and cosine distance classfication head!"
        z_support, z_query, y_support, y_query  = self.parse_feature(x,is_feature)
        self.feat_dim = z_support.size()[1]

        if self.cls_head_type == 'fc':
            self.cls_head = nn.Linear(self.feat_dim, self.n_way)
        if self.cls_head_type == 'fc2':
            self.cls_head = nn.Sequential(
                            nn.Linear(self.feat_dim, self.fc_inter_dim),
                            nn.ReLU(inplace=True),
                            nn.Linear(self.fc_inter_dim, self.n_way)
                            )
        elif self.cls_head_type == 'cosine_dist':        
            self.cls_head = cosineDist(self.feat_dim, self.n_way)

        self.cls_head.to(self.device)

        set_optimizer = torch.optim.SGD(self.cls_head.parameters(), lr = 0.01, momentum=0.9, dampening=0.9, weight_decay=0.001)

        loss_function = nn.CrossEntropyLoss()
        # loss_function = loss_function.cuda()
        
        batch_size = self.batch_size
        support_size = self.n_way* self.n_support
        self.cls_head.train()

        for epoch in range(self.num_epochs):
            rand_id = np.random.permutation(support_size)
            for i in range(0, support_size, batch_size):
                set_optimizer.zero_grad()
                selected_id = torch.from_numpy( rand_id[i: min(i+batch_size, support_size) ]).to(self.device)
                z_batch = z_support[selected_id]
                y_batch = y_support[selected_id] 
                scores = self.cls_head(z_batch)
                loss = loss_function(scores,y_batch)
                loss.backward()
                set_optimizer.step()
        
        th = self.threshold 
        self.cls_head.eval()
        with torch.no_grad():
          scores_support = self.cls_head(z_support)
          pred_tr = torch.argmax((torch.softmax(scores_support,-1)>th).float(), -1).cpu().numpy()
          train_acc = np.mean(pred_tr == y_support.cpu().numpy())
          
          scores_query = self.cls_head(z_query)
          pred_eval = torch.argmax((torch.softmax(scores_query,-1)>th).float(), -1).cpu().numpy()
          eval_acc = np.mean(pred_eval == y_query.cpu().numpy())

        return eval_acc, train_acc, pred_eval

    def set_forward_adaptation_video(self, support_dataset, query_dataset, mode='mean'):
        assert self.cls_head_type in ['fc', 'fc2', 'cosine_dist'], "forward adaptation only support fc and cosine distance classfication head!"
        support_size = len(support_dataset)
        inputs_support = []
        target_support = []
        #inputs dim [np, nf]
        for inputs, target, _ in support_dataset:
            inputs = torch.tensor(inputs)
            index = inputs.size()[0] // 2
            inputs_support.append(inputs[index].unsqueeze(0))
            target_support.append(target)

        self.feat_dim = inputs.size()[1]

        if self.cls_head_type == 'fc':
            self.cls_head = nn.Linear(self.feat_dim, self.n_way)
        if self.cls_head_type == 'fc2':
            self.cls_head = nn.Sequential(
                            nn.Linear(self.feat_dim, self.fc_inter_dim),
                            nn.ReLU(inplace=True),
                            nn.Linear(self.fc_inter_dim, self.n_way)
                            )
        elif self.cls_head_type == 'cosine_dist':        
            self.cls_head = cosineDist(self.feat_dim, self.n_way)

        self.cls_head.to(self.device)

        set_optimizer = torch.optim.SGD(self.cls_head.parameters(), lr = 0.01, momentum=0.9, dampening=0.9, weight_decay=0.001)

        loss_function = nn.CrossEntropyLoss()
        # loss_function = loss_function.cuda()
        
        batch_size = self.batch_size
        self.cls_head.train()
        for epoch in range(self.num_epochs):
            rand_id = np.random.permutation(support_size)
            for i in range(0, support_size, batch_size):
                selected_id = rand_id[i: min(i+batch_size, support_size)]
                scores_batch = []
                target_batch = []
                for idx in selected_id:
                    inputs = inputs_support[idx]
                    target = target_support[idx]
                    scores = self.cls_head(inputs.to(self.device))
                    scores_batch.append(scores)
                    target_batch.append(target)
                set_optimizer.zero_grad()
                scores_batch = torch.cat(scores_batch)
                target_batch = torch.tensor(target_batch)
                loss = loss_function(scores_batch, target_batch.to(self.device))
                loss.backward()
                set_optimizer.step()
        
        th = self.threshold 
        self.cls_head.eval()
        scores_query = []
        target_query = []

        scores_support = []
        with torch.no_grad():
            for i, (inputs, target, _) in enumerate(query_dataset): 
                inputs = torch.tensor(inputs)
                scores = self.cls_head(inputs.to(self.device))
                if mode == 'mean':
                    scores = torch.mean(scores, dim=0).unsqueeze(0)
                else:
                    scores = torch.max(scores, dim=0)[0].unsqueeze(0)
                scores_query.append(scores)
                target_query.append(target)
            
            for i in range(support_size): 
                inputs = inputs_support[i]
                scores = self.cls_head(inputs.to(self.device))
                scores_support.append(scores)

        scores_support = torch.cat(scores_support)         
        scores_query = torch.cat(scores_query) 
        pred_eval = torch.argmax((torch.softmax(scores_query,-1)>th).float(), -1).cpu().numpy()
        pred_tr = torch.argmax((torch.softmax(scores_support,-1)>th).float(), -1).cpu().numpy()     
        eval_acc = np.mean(pred_eval == target_query)
        train_acc = np.mean(pred_tr == target_support)

        return eval_acc, train_acc, pred_eval, target_query
    
    def parse_feature(self,x,is_feature):
        if is_feature:
            z_support = x['z_support'].to(self.device)
            z_query = x['z_query'].to(self.device)
            y_support = x['support_labels'].to(self.device)
            y_query = x['query_labels'].to(self.device)

        else:
            assert is_feature == True, 'Baseline only support testing with features'

        return z_support, z_query, y_support, y_query


class BaselineFinetune_DA():
    def __init__(self, 
        device,
        cls_head_type, 
        arch='rgb_r2plus1d_8f_34_encoder',
        modelLocation='',
        saved_model_name='model_best',
        length=8,
        image_size=224,
        feat_dim=512,
        n_way=2, 
        n_support=5, 
        n_query=15, 
        batch_size=4,
        num_epochs=301,
        threshold=0.5,
        fc_inter_dim=128,
        # temperature=128,
        freeze_backbone=True,
        train_full_encoder=True,
        ):
        self.device = device
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = n_query
        self.cls_head_type = cls_head_type
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.threshold = threshold 
        # self.temperature = temperature
        self.freeze_backbone = freeze_backbone
        self.train_full_encoder = train_full_encoder
        self.length = length
        self.image_size = image_size
        self.arch = arch
        self.modelLocation = modelLocation
        self.saved_model_name = saved_model_name
        self.feat_dim = feat_dim
        self.fc_inter_dim = fc_inter_dim


    def label_prop(self, x_lp, y_lp, n_way, k_lp=10, delta=0.2, alpha=0.5):
        neigh = NearestNeighbors(n_neighbors=k_lp)
        neigh.fit(x_lp)
        d_lp, idx_lp = neigh.kneighbors(x_lp)
        d_lp = np.power(d_lp, 2)
        sigma2_lp = np.mean(d_lp)
        n_lp = len(y_lp)
        del_n = int(n_lp * (1.0 - delta))
        for i in range(n_way):
            yi = y_lp[:, i]
            top_del_idx = np.argsort(yi)[0:del_n]
            y_lp[top_del_idx, i] = 0
            
        w_lp = np.zeros((n_lp, n_lp))
        for i in range(n_lp):
            for j in range(k_lp):
                xj = idx_lp[i, j]
                w_lp[i, xj] = np.exp(-d_lp[i, j] / (2 * sigma2_lp))
                w_lp[xj, i] = np.exp(-d_lp[i, j] / (2 * sigma2_lp))
        q_lp = np.diag(np.sum(w_lp, axis=1))
        q2_lp = sqrtm(q_lp)
        q2_lp = np.linalg.inv(q2_lp)
        L_lp = np.matmul(np.matmul(q2_lp, w_lp), q2_lp)
        a_lp = np.eye(n_lp) - alpha * L_lp
        a_lp = np.linalg.inv(a_lp)
        # return the new predictions
        return np.matmul(a_lp, y_lp)

    
        # inputs is a list of tuples (x, y) with length num augs
    
    # inputs is a list of tuples (x, y) with length num augs
    # x is a list of images with length seq length, and dim [n_way * (n_support + n_query) channles img_size img_size]
    def set_forward_adaptation(self, inputs, do_lp=False, k_lp=10, delta=0.45, alpha=0.5):
        assert self.cls_head_type in ['fc', 'fc2', 'cosine_dist'], "forward adaptation only support fc and cosine distance classfication head!"

        self.encoder = self.load_encoder(self.arch, self.modelLocation, self.saved_model_name)
        self.encoder_without_dp = self.encoder
        if self.device.type == 'cuda' and torch.cuda.device_count() > 1:
            # print("use data parallel")
            self.encoder = torch.nn.DataParallel(self.encoder) 
            self.encoder_without_dp = self.encoder.module

        if self.cls_head_type == 'fc':
            self.cls_head= nn.Linear(self.feat_dim, self.n_way)
        if self.cls_head_type == 'fc2':
            self.cls_head = nn.Sequential(
                            nn.Linear(self.feat_dim, self.fc_inter_dim),
                            nn.ReLU(inplace=True),
                            nn.Linear(self.fc_inter_dim, self.n_way)
                            )
        elif self.cls_head_type == 'cosine_dist':        
            self.cls_head = cosineDist(self.feat_dim, self.n_way)
        self.avgpool = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
        self.cls_head.to(self.device)
        self.encoder.to(self.device)

        support_size = self.n_way * self.n_support

        liz_x = [torch.cat(x, dim=1).view(-1,self.length,3,self.image_size,self.image_size).transpose(1,2) for (x,y) in inputs]
        liz_y = [y for (x,y) in inputs]
        x = liz_x[0]
        y = liz_y[0]
      
        y_s_i = y[:support_size]
        y_q_i = y[support_size:]

        x_s_i = x[:support_size]
        x_q_i = x[support_size:]

        x_s_i = torch.cat((x_s_i, x_s_i, x_s_i), dim = 0) ##oversample the first one
        y_s_i = torch.cat((y_s_i, y_s_i, y_s_i), dim = 0)
        for x_aug, y_aug in zip(liz_x[1:], liz_y[1:]):
            x_s_aug = x_aug[:support_size]
            y_s_aug = y_aug[:support_size]
            x_s_i = torch.cat((x_s_i, x_s_aug), dim = 0)
            y_s_i = torch.cat((y_s_i, y_s_aug), dim = 0)

        lengt = len(liz_x) + 2 #oversampled
        loss_fn = nn.CrossEntropyLoss()

        classifier_opt = torch.optim.SGD(self.cls_head.parameters(), lr = 0.01, momentum=0.9, dampening=0.9, weight_decay=0.001)
        #torch.optim.Adam(cls_head.parameters(), lr = 0.01, weight_decay=0.001)
        if self.freeze_backbone is False:
            # print("update encoder")
            if self.train_full_encoder is False:
                names = []
                for name, param in self.encoder.named_parameters():
                    if param.requires_grad:
                        #print(name)
                        names.append(name)
                names_sub = names[:-12] ### last R2plus1d block can adapt

                for name, param in self.encoder.named_parameters():
                    if name in names_sub:
                        param.requires_grad = False    
            delta_opt = torch.optim.Adam(filter(lambda p: p.requires_grad, self.encoder.parameters()), lr = 0.0001)
        else:
            for param in self.encoder.parameters():
                param.requires_grad = False

        batch_size = self.batch_size
        
        if self.freeze_backbone is False:
            self.encoder.train()
        else:
            self.encoder.eval()
        self.cls_head.train()

        for epoch in range(self.num_epochs):
            # print("epoch: {}".format(epoch))
            rand_id = np.random.permutation(support_size * lengt)
            for j in range(0, support_size * lengt, batch_size):
                classifier_opt.zero_grad()
                if self.freeze_backbone is False:
                    delta_opt.zero_grad()
                selected_id = torch.from_numpy( rand_id[j: min(j+batch_size, support_size * lengt)])
                z_batch = x_s_i[selected_id]
                y_batch = y_s_i[selected_id]
                z_batch = z_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                features = self.encoder(z_batch)
                features = self.avgpool(features).squeeze()
                if features.dim() < 2:
                    features = features.unsqueeze(0)
                scores = self.cls_head(features)
                loss = loss_fn(scores, y_batch)

                loss.backward()
                classifier_opt.step()
                if self.freeze_backbone is False:
                    delta_opt.step()
                    # print(loss.item())
        
        th = self.threshold 
        self.cls_head.eval()
        self.encoder.eval()
        #only test on the unauged smaples
        with torch.no_grad():
            x_s = liz_x[0][:support_size].to(self.device)
            y_s = liz_y[0][:support_size].to(self.device)
            x_q = liz_x[0][support_size:].to(self.device)
            y_q = liz_y[0][support_size:].to(self.device)
            
            z_support = self.encoder(x_s)
            z_support = self.avgpool(z_support).squeeze()
            if z_support.dim() < 2:
                z_support = z_support.unsqueeze(0)
            scores_support = self.cls_head(z_support)
            
            z_query = self.encoder(x_q)
            z_query = self.avgpool(z_query).squeeze()
            if z_query.dim() < 2:
                z_query = z_query.unsqueeze(0)
            scores_query = self.cls_head(z_query)
        
        pred_tr = torch.argmax((torch.softmax(scores_support,-1)>th).float(), -1).cpu().numpy()
        train_acc = np.mean(pred_tr == y_s.cpu().numpy())
        
        if not do_lp:     
            pred_eval = torch.argmax((torch.softmax(scores_query,-1)>th).float(), -1).cpu().numpy()
            eval_acc = np.mean(pred_eval == y_q.cpu().numpy())
        else:
            x_lp = z_query.cpu().numpy()
            y_lp = torch.softmax(scores_query, 1).cpu().numpy()
            ynew_lp = self.label_prop(x_lp, y_lp, self.n_way, k_lp, delta, alpha)
            pred_eval = np.argmax((ynew_lp>th), -1)
            eval_acc = np.mean(pred_eval == y_q.cpu().numpy())

        return eval_acc, train_acc, pred_eval, y_q.cpu().numpy()
    
    def load_encoder(self, arch, modelLocation, saved_model_name):
        encoder=models.__dict__[arch](modelPath='')
        if modelLocation:
            model_path = os.path.join(modelLocation,saved_model_name + '.pth.tar') 
            params = torch.load(model_path)
            # print(modelLocation)
            encoder.load_state_dict(params['state_dict_encoder'])
        return encoder


class BaselineFinetune_DA2():
    def __init__(self, 
        device,
        cls_head_type, 
        arch='rgb_r2plus1d_8f_34_encoder',
        modelLocation='',
        saved_model_name='model_best',
        length=8,
        image_size=224,
        feat_dim=512,
        n_way=2, 
        n_support=5, 
        n_query=15, 
        batch_size=4,
        num_epochs=301,
        threshold=0.5,
        fc_inter_dim=128,
        # temperature=128,
        freeze_backbone=True,
        train_full_encoder=True,
        ):
        self.device = device
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = n_query
        self.cls_head_type = cls_head_type
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.threshold = threshold 
        # self.temperature = temperature
        self.freeze_backbone = freeze_backbone
        self.train_full_encoder = train_full_encoder
        self.length = length
        self.image_size = image_size
        self.arch = arch
        self.modelLocation = modelLocation
        self.saved_model_name = saved_model_name
        self.feat_dim = feat_dim
        self.fc_inter_dim = fc_inter_dim


    def label_prop(self, x_lp, y_lp, n_way, k_lp=10, delta=0.2, alpha=0.5):
        neigh = NearestNeighbors(n_neighbors=k_lp)
        neigh.fit(x_lp)
        d_lp, idx_lp = neigh.kneighbors(x_lp)
        d_lp = np.power(d_lp, 2)
        sigma2_lp = np.mean(d_lp)
        n_lp = len(y_lp)
        del_n = int(n_lp * (1.0 - delta))
        for i in range(n_way):
            yi = y_lp[:, i]
            top_del_idx = np.argsort(yi)[0:del_n]
            y_lp[top_del_idx, i] = 0
            
        w_lp = np.zeros((n_lp, n_lp))
        for i in range(n_lp):
            for j in range(k_lp):
                xj = idx_lp[i, j]
                w_lp[i, xj] = np.exp(-d_lp[i, j] / (2 * sigma2_lp))
                w_lp[xj, i] = np.exp(-d_lp[i, j] / (2 * sigma2_lp))
        q_lp = np.diag(np.sum(w_lp, axis=1))
        q2_lp = sqrtm(q_lp)
        q2_lp = np.linalg.inv(q2_lp)
        L_lp = np.matmul(np.matmul(q2_lp, w_lp), q2_lp)
        a_lp = np.eye(n_lp) - alpha * L_lp
        a_lp = np.linalg.inv(a_lp)
        # return the new predictions
        return np.matmul(a_lp, y_lp)

    
        # inputs is a list of tuples (x, y) with length num augs
    
    # inputs is a list of tuples (x, y) with length num augs
    # x is a list of images with length seq length, and dim [n_way * (n_support + n_query) channles img_size img_size]
    def set_forward_adaptation(self, inputs, do_lp=False, k_lp=10, delta=0.45, alpha=0.5):
        assert self.cls_head_type in ['fc', 'fc2', 'cosine_dist'], "forward adaptation only support fc and cosine distance classfication head!"

        self.encoder = self.load_encoder(self.arch, self.modelLocation, self.saved_model_name)
        self.encoder_without_dp = self.encoder
        if self.device.type == 'cuda' and torch.cuda.device_count() > 1:
            # print("use data parallel")
            self.encoder = torch.nn.DataParallel(self.encoder) 
            self.encoder_without_dp = self.encoder.module

        if self.cls_head_type == 'fc':
            self.cls_head= nn.Linear(self.feat_dim, self.n_way)
        if self.cls_head_type == 'fc2':
            self.cls_head = nn.Sequential(
                            nn.Linear(self.feat_dim, self.fc_inter_dim),
                            nn.ReLU(inplace=True),
                            nn.Linear(self.fc_inter_dim, self.n_way)
                            )
        elif self.cls_head_type == 'cosine_dist':        
            self.cls_head = cosineDist(self.feat_dim, self.n_way)
        self.avgpool = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
        self.cls_head.to(self.device)
        self.encoder.to(self.device)

        support_size = self.n_way * self.n_support

        liz_x = [torch.cat(x, dim=1).view(-1,self.length,3,self.image_size,self.image_size).transpose(1,2) for (x,y) in inputs]
        liz_y = [y for (x,y) in inputs]
        x = liz_x[0]
        y = liz_y[0]
      
        y_s_i = y[:support_size]
        y_q_i = y[support_size:]

        x_s_i = x[:support_size]
        x_q_i = x[support_size:]

        x_s_i = torch.cat((x_s_i, x_s_i, x_s_i), dim = 0) ##oversample the first one
        y_s_i = torch.cat((y_s_i, y_s_i, y_s_i), dim = 0)
        for x_aug, y_aug in zip(liz_x[1:], liz_y[1:]):
            x_s_aug = x_aug[:support_size]
            y_s_aug = y_aug[:support_size]
            x_s_i = torch.cat((x_s_i, x_s_aug), dim = 0)
            y_s_i = torch.cat((y_s_i, y_s_aug), dim = 0)

        lengt = len(liz_x) + 2 #oversampled
        loss_fn = nn.CrossEntropyLoss()

        classifier_opt = torch.optim.SGD(self.cls_head.parameters(), lr = 0.01, momentum=0.9, dampening=0.9, weight_decay=0.001)
        # classifier_opt = torch.optim.Adam(self.cls_head.parameters(), lr = 0.01, weight_decay=0.001)
        if self.freeze_backbone is False:
            # print("update encoder")
            if self.train_full_encoder is False:
                names = []
                for name, param in self.encoder.named_parameters():
                    if param.requires_grad:
                        #print(name)
                        names.append(name)
                names_sub = names[:-12] ### last R2plus1d block can adapt

                for name, param in self.encoder.named_parameters():
                    if name in names_sub:
                        param.requires_grad = False    
            delta_opt = torch.optim.Adam(filter(lambda p: p.requires_grad, self.encoder.parameters()), lr = 0.00001)
            # delta_opt = torch.optim.SGD(filter(lambda p: p.requires_grad, self.encoder.parameters()), lr = 0.001, momentum=0.9, dampening=0.9, weight_decay=0.01)
        else:
            for param in self.encoder.parameters():
                param.requires_grad = False

        batch_size = self.batch_size
        
        if self.freeze_backbone is False:
            self.encoder.train()
        else:
            self.encoder.eval()
        self.cls_head.train()

        for epoch in range(self.num_epochs):
            print("epoch: {}".format(epoch))
            rand_id = np.random.permutation(support_size * lengt)
            for j in range(0, support_size * lengt, batch_size):
                classifier_opt.zero_grad()
                if self.freeze_backbone is False:
                    delta_opt.zero_grad()
                selected_id = torch.from_numpy( rand_id[j: min(j+batch_size, support_size * lengt)])
                z_batch = x_s_i[selected_id]
                y_batch = y_s_i[selected_id]
                z_batch = z_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                features = self.encoder(z_batch)
                features = self.avgpool(features).squeeze()
                if features.dim() < 2:
                    features = features.unsqueeze(0)
                scores = self.cls_head(features)
                loss = loss_fn(scores, y_batch)

                loss.backward()
                classifier_opt.step()
                if self.freeze_backbone is False:
                    delta_opt.step()
                    print(loss.item())
        
        th = self.threshold 
        self.cls_head.eval()
        self.encoder.eval()
        #only test on the unauged smaples
        with torch.no_grad():
            x_s = liz_x[0][:support_size].to(self.device)
            y_s = liz_y[0][:support_size].to(self.device)
            x_q = liz_x[0][support_size:].to(self.device)
            y_q = liz_y[0][support_size:].to(self.device)
            
            z_support = self.encoder(x_s)
            z_support = self.avgpool(z_support).squeeze()
            if z_support.dim() < 2:
                z_support = z_support.unsqueeze(0)
            scores_support = self.cls_head(z_support)
            
            z_query = self.encoder(x_q)
            z_query = self.avgpool(z_query).squeeze()
            if z_query.dim() < 2:
                z_query = z_query.unsqueeze(0)
            scores_query = self.cls_head(z_query)
        
        pred_tr = torch.argmax((torch.softmax(scores_support,-1)>th).float(), -1).cpu().numpy()
        train_acc = np.mean(pred_tr == y_s.cpu().numpy())
        
        if not do_lp:     
            pred_eval = torch.argmax((torch.softmax(scores_query,-1)>th).float(), -1).cpu().numpy()
            eval_acc = np.mean(pred_eval == y_q.cpu().numpy())
        else:
            x_lp = z_query.cpu().numpy()
            y_lp = torch.softmax(scores_query, 1).cpu().numpy()
            ynew_lp = self.label_prop(x_lp, y_lp, self.n_way, k_lp, delta, alpha)
            pred_eval = np.argmax((ynew_lp>th), -1)
            eval_acc = np.mean(pred_eval == y_q.cpu().numpy())

        return eval_acc, train_acc, pred_eval, y_q.cpu().numpy()
    
    def load_encoder(self, arch, modelLocation, saved_model_name):
        encoder=models.__dict__[arch](modelPath='')
        if modelLocation:
            model_path = os.path.join(modelLocation,saved_model_name + '.pth.tar') 
            params = torch.load(model_path)
            # print(modelLocation)
            encoder.load_state_dict(params['state_dict_encoder'])
        return encoder



class BaselineFinetune_DA3():
    def __init__(self, 
        device,
        cls_head_type, 
        arch='rgb_r2plus1d_8f_34_encoder',
        modelLocation='',
        saved_model_name='model_best',
        length=8,
        image_size=224,
        feat_dim=512,
        n_way=2, 
        n_support=5, 
        n_query=15, 
        batch_size=4,
        num_epochs=301,
        threshold=0.5,
        # temperature=128,
        freeze_backbone=True,
        train_full_encoder=True,
        ):
        self.device = device
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = n_query
        self.cls_head_type = cls_head_type
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.threshold = threshold 
        # self.temperature = temperature
        self.freeze_backbone = freeze_backbone
        self.train_full_encoder = train_full_encoder
        self.length = length
        self.image_size = image_size
        self.arch = arch
        self.modelLocation = modelLocation
        self.saved_model_name = saved_model_name
        self.feat_dim = feat_dim


    def label_prop(self, x_lp, y_lp, n_way, k_lp=10, delta=0.2, alpha=0.5):
        neigh = NearestNeighbors(n_neighbors=k_lp)
        neigh.fit(x_lp)
        d_lp, idx_lp = neigh.kneighbors(x_lp)
        d_lp = np.power(d_lp, 2)
        sigma2_lp = np.mean(d_lp)
        n_lp = len(y_lp)
        del_n = int(n_lp * (1.0 - delta))
        for i in range(n_way):
            yi = y_lp[:, i]
            top_del_idx = np.argsort(yi)[0:del_n]
            y_lp[top_del_idx, i] = 0
            
        w_lp = np.zeros((n_lp, n_lp))
        for i in range(n_lp):
            for j in range(k_lp):
                xj = idx_lp[i, j]
                w_lp[i, xj] = np.exp(-d_lp[i, j] / (2 * sigma2_lp))
                w_lp[xj, i] = np.exp(-d_lp[i, j] / (2 * sigma2_lp))
        q_lp = np.diag(np.sum(w_lp, axis=1))
        q2_lp = sqrtm(q_lp)
        q2_lp = np.linalg.inv(q2_lp)
        L_lp = np.matmul(np.matmul(q2_lp, w_lp), q2_lp)
        a_lp = np.eye(n_lp) - alpha * L_lp
        a_lp = np.linalg.inv(a_lp)
        # return the new predictions
        return np.matmul(a_lp, y_lp)

    
        # inputs is a list of tuples (x, y) with length num augs
    
    # inputs is a list of tuples (x, y) with length num augs
    # x is a list of images with length seq length, and dim [n_way * (n_support + n_query) channles img_size img_size]
    def set_forward_adaptation(self, inputs, do_lp=False, k_lp=10, delta=0.45, alpha=0.5):
        assert self.cls_head_type in ['fc', 'cosine_dist'], "forward adaptation only support fc and cosine distance classfication head!"

        self.encoder = self.load_encoder(self.arch, self.modelLocation, self.saved_model_name)
        self.encoder_without_dp = self.encoder
        if self.device.type == 'cuda' and torch.cuda.device_count() > 1:
            # print("use data parallel")
            self.encoder = torch.nn.DataParallel(self.encoder) 
            self.encoder_without_dp = self.encoder.module

        if self.cls_head_type == 'fc':
            self.cls_head= nn.Linear(self.feat_dim, self.n_way)
        elif self.cls_head_type == 'cosine_dist':        
            self.cls_head = cosineDist(self.feat_dim, self.n_way)
        self.avgpool = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
        self.cls_head.to(self.device)
        self.encoder.to(self.device)

        support_size = self.n_way * self.n_support

        liz_x = [torch.cat(x, dim=1).view(-1,self.length,3,self.image_size,self.image_size).transpose(1,2) for (x,y) in inputs]
        liz_y = [y for (x,y) in inputs]
        x = liz_x[0]
        y = liz_y[0]
      
        y_s_i = y[:support_size]
        y_q_i = y[support_size:]

        x_s_i = x[:support_size]
        x_q_i = x[support_size:]

        x_s_i = torch.cat((x_s_i, x_s_i, x_s_i), dim = 0) ##oversample the first one
        y_s_i = torch.cat((y_s_i, y_s_i, y_s_i), dim = 0)
        for x_aug, y_aug in zip(liz_x[1:], liz_y[1:]):
            x_s_aug = x_aug[:support_size]
            y_s_aug = y_aug[:support_size]
            x_s_i = torch.cat((x_s_i, x_s_aug), dim = 0)
            y_s_i = torch.cat((y_s_i, y_s_aug), dim = 0)

        lengt = len(liz_x) + 2 #oversampled
        loss_fn = nn.CrossEntropyLoss()

        classifier_opt = torch.optim.SGD(self.cls_head.parameters(), lr = 0.01, momentum=0.9, dampening=0.9, weight_decay=0.001)
        # classifier_opt = torch.optim.Adam(self.cls_head.parameters(), lr = 0.01, weight_decay=0.001)
        if self.freeze_backbone is False:
            # print("update encoder")
            if self.train_full_encoder is False:
                names = []
                for name, param in self.encoder.named_parameters():
                    if param.requires_grad:
                        #print(name)
                        names.append(name)
                names_sub = names[:-12] ### last R2plus1d block can adapt

                for name, param in self.encoder.named_parameters():
                    if name in names_sub:
                        param.requires_grad = False    
            delta_opt = torch.optim.Adam(filter(lambda p: p.requires_grad, self.encoder.parameters()), lr = 0.0001)
        else:
            for param in self.encoder.parameters():
                param.requires_grad = False

        batch_size = self.batch_size
        
        if self.freeze_backbone is False:
            self.encoder.train()
        else:
            self.encoder.eval()
        self.cls_head.train()

        for epoch in range(self.num_epochs):
            print("epoch: {}".format(epoch))
            rand_id = np.random.permutation(support_size * lengt)
            for j in range(0, support_size * lengt, batch_size):
                classifier_opt.zero_grad()
                if self.freeze_backbone is False:
                    delta_opt.zero_grad()
                selected_id = torch.from_numpy( rand_id[j: min(j+batch_size, support_size * lengt)])
                z_batch = x_s_i[selected_id]
                y_batch = y_s_i[selected_id]
                z_batch = z_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                features = self.encoder(z_batch)
                features = self.avgpool(features).squeeze()
                if features.dim() < 2:
                    features = features.unsqueeze(0)
                scores = self.cls_head(features)
                loss = loss_fn(scores, y_batch)

                loss.backward()
                classifier_opt.step()
                if self.freeze_backbone is False:
                    delta_opt.step()
                    print(loss.item())
        
        th = self.threshold 
        self.cls_head.eval()
        self.encoder.eval()
        #only test on the unauged smaples
        with torch.no_grad():
            x_s = liz_x[0][:support_size].to(self.device)
            y_s = liz_y[0][:support_size].to(self.device)
            x_q = liz_x[0][support_size:].to(self.device)
            y_q = liz_y[0][support_size:].to(self.device)
            
            z_support = self.encoder(x_s)
            z_support = self.avgpool(z_support).squeeze()
            if z_support.dim() < 2:
                z_support = z_support.unsqueeze(0)
            scores_support = self.cls_head(z_support)
            
            z_query = self.encoder(x_q)
            z_query = self.avgpool(z_query).squeeze()
            if z_query.dim() < 2:
                z_query = z_query.unsqueeze(0)
            scores_query = self.cls_head(z_query)
        
        pred_tr = torch.argmax((torch.softmax(scores_support,-1)>th).float(), -1).cpu().numpy()
        train_acc = np.mean(pred_tr == y_s.cpu().numpy())
        
        if not do_lp:     
            pred_eval = torch.argmax((torch.softmax(scores_query,-1)>th).float(), -1).cpu().numpy()
            eval_acc = np.mean(pred_eval == y_q.cpu().numpy())
        else:
            x_lp = z_query.cpu().numpy()
            y_lp = torch.softmax(scores_query, 1).cpu().numpy()
            ynew_lp = self.label_prop(x_lp, y_lp, self.n_way, k_lp, delta, alpha)
            pred_eval = np.argmax((ynew_lp>th), -1)
            eval_acc = np.mean(pred_eval == y_q.cpu().numpy())

        return eval_acc, train_acc, pred_eval, y_q.cpu().numpy()
    
    def load_encoder(self, arch, modelLocation, saved_model_name):
        encoder=models.__dict__[arch](modelPath='')
        if modelLocation:
            model_path = os.path.join(modelLocation,saved_model_name + '.pth.tar') 
            params = torch.load(model_path)
            # print(modelLocation)
            encoder.load_state_dict(params['state_dict_encoder'])
        return encoder