import numpy as np
import torch 
import torch.nn as nn
from .classification_heads import protonet, ClassificationHead, cosineDist

class BaselineFinetune():
    def __init__(self, 
        device,
        n_way, 
        n_support, 
        cls_head_type, 
        n_query=None, 
        batch_size=4,
        threshold=0.5,
        temperature=128
        ):
        self.device = device
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = n_query
        self.cls_head_type = cls_head_type
        self.batch_size = batch_size
        self.threshold = threshold 
        self.temperature = temperature
        
    def set_forward(self,x,is_feature = True):
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
        
        th = self.threshold
        pred_tr = torch.argmax((torch.softmax(scores_support,-1)>th).float(), -1).cpu().numpy()
        train_acc = np.mean(pred_tr == y_support.cpu().numpy())
        
        pred_eval = torch.argmax((torch.softmax(scores_query,-1)>th).float(), -1).cpu().numpy()
        eval_acc = np.mean(pred_eval == y_query.cpu().numpy())

        return eval_acc, train_acc, pred_eval 
 
    def set_forward_adaptation(self,x,is_feature = True):
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

        for epoch in range(301):
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

    def parse_feature(self,x,is_feature):
        if is_feature:
            z_support = x['z_support'].to(self.device)
            z_query = x['z_query'].to(self.device)
            y_support = x['support_labels'].to(self.device)
            y_query = x['query_labels'].to(self.device)

        else:
            assert is_feature == True, 'Baseline only support testing with features'

        return z_support, z_query, y_support, y_query
