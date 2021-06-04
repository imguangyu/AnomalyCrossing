import torch
import numpy as np

# label is a list on integetrs indicated the label of the samples, starts from 0
# n_batch is the number of tasks in an episode, e.g. 100
# n_cls is the num of classes or the nums of ways
# n_per is the num of samples per batch, i.e. num of query + num of support
# so for each patch the batch size if n_per * n_cls
class CategoriesSampler():

    def __init__(self, label, n_batch, n_cls, n_per):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per
        label_set = set(label)
        label = np.array(label)
        self.m_ind = []
        for i in label_set:
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]
            for c in classes:
                l = self.m_ind[c]
                pos = torch.randperm(len(l))[:self.n_per]
                batch.append(l[pos])
            # Note this transpose is very important 
            # it determines how we should generate labels in training 
            # before transpose the batch matrix is w by (s+q)
            # after transpose the dim is (s+q) + w 
            # so the flatten is go through each classes first 
            # so the labels should be range(w) * repeat(q)
            batch = torch.stack(batch).t().reshape(-1)
            yield batch

class CategoriesSampler2():
    def __init__(self, label, n_batch, n_cls, n_per):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per
        label_set = set(label)
        label = np.array(label)
        self.m_ind = []
        for i in label_set:
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch
    
    def generate_perm(self):
        self.perms = []
        for i_batch in range(self.n_batch):
            batch = []
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]
            for c in classes:
                l = self.m_ind[c]
                pos = torch.randperm(len(l))[:self.n_per]
                batch.append(l[pos])
            # Note this transpose is very important 
            # it determines how we should generate labels in training 
            # before transpose the batch matrix is w by (s+q)
            # after transpose the dim is (s+q) + w 
            # so the flatten is go through each classes first 
            # so the labels should be range(w) * repeat(q)
            self.perms.append(torch.stack(batch).t().reshape(-1))
    def __iter__(self):
        for i in range(self.n_batch):
            yield self.perms[i]