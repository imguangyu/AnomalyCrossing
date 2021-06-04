import torch
import numpy as np
import h5py

class SimpleHDF5Dataset:
    def __init__(self, file_handle = None):
        if file_handle == None:
            self.f = ''
            self.all_feats_dset = []
            self.all_labels = []
            # self.all_img_paths = []
            self.total = 0 
        else:
            self.f = file_handle
            self.all_feats_dset = self.f['all_feats'][...]
            self.all_labels = self.f['all_labels'][...]
            # self.all_img_paths = self.f['all_img_paths'][...]
            self.total = self.f['count'][0]
    def __getitem__(self, i):
        return torch.Tensor(self.all_feats_dset[i,:]), int(self.all_labels[i]), #self.all_img_paths[i]

    def __len__(self):
        return self.total

def init_loader(filename):
    with h5py.File(filename, 'r') as f:
        fileset = SimpleHDF5Dataset(f)

    #labels = [ l for l  in fileset.all_labels if l != 0]
    feats = fileset.all_feats_dset
    labels = fileset.all_labels
    # img_paths = fileset.all_img_paths
    while np.sum(feats[-1]) == 0:
        feats  = np.delete(feats,-1,axis = 0)
        labels = np.delete(labels,-1,axis = 0)
        # img_paths = np.delete(img_paths,-1,axis = 0)
        
    class_list = np.unique(np.array(labels)).tolist() 
    inds = range(len(labels))

    cl_data_file = {}
    # cl_path_file = {}
    for cl in class_list:
        cl_data_file[cl] = []
        # cl_path_file[cl] = []
    for ind in inds:
        cl_data_file[labels[ind]].append( feats[ind])
        # cl_path_file[labels[ind]].append(img_paths[ind].decode("utf-8"))

    return cl_data_file#, cl_path_file