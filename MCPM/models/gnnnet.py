import torch
import torch.nn as nn
import numpy as np
# from methods.meta_template import MetaTemplate
from .gnn import GNN_nl
from torch.autograd import Variable
import models.backbone as backbone
import copy


class Classifier(nn.Module):
    def __init__(self, dim, n_way):
        super(Classifier, self).__init__()
        
        self.fc = nn.Linear(dim, n_way)

    def forward(self, x):
        x = self.fc(x)
        return x

class GnnNet(nn.Module):
  maml=False
  def __init__(self, encoder, feat_dim, n_way, n_support, n_query, 
                     gnn_node_dim=128, gnn_nf=86, 
                     multi_gpu=True, first=True):
    super(GnnNet, self).__init__()

    # loss function
    # self.loss_fn = nn.CrossEntropyLoss()
    self.first = first
    self.multi_gpu = multi_gpu
    
    self.encoder = encoder
    self.encoder_without_dp = self.encoder
    self.feat_dim = feat_dim
    self.n_way      = n_way
    self.n_support  = n_support
    self.n_query    = n_query
    
    if self.multi_gpu and torch.cuda.device_count() > 1:
      self.encoder = torch.nn.DataParallel(self.encoder)
      self.encoder_without_dp = self.encoder.module
    
    self.avgpool = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
    # metric function
    # feat_dim 512 Resnet10
    self.gnn_node_dim = gnn_node_dim
    self.gnn_nf = gnn_nf
    self.fc = nn.Sequential(nn.Linear(self.feat_dim, self.gnn_node_dim), nn.BatchNorm1d(self.gnn_node_dim, track_running_stats=False)) if not self.maml else nn.Sequential(backbone.Linear_fw(self.feat_dim, self.gnn_node_dim), backbone.BatchNorm1d_fw(self.gnn_node_dim, track_running_stats=False))
    self.gnn = GNN_nl(self.gnn_node_dim + self.n_way, self.gnn_nf, self.n_way)
    self.method = 'GnnNet'

    # fix label for training the metric function   1*nw(1 + ns)*nw
    support_label = torch.from_numpy(np.repeat(range(self.n_way), self.n_support)).unsqueeze(1)
    # one hot version of the support labels
    support_label = torch.zeros(self.n_way*self.n_support, self.n_way).scatter(1, support_label, 1).view(self.n_way, self.n_support, self.n_way)
    # it seems that here it is assumed that the num of query is 1 
    # append it on the support_label
    support_label = torch.cat([support_label, torch.zeros(self.n_way, 1, n_way)], dim=1)
    self.support_label = support_label.view(1, -1, self.n_way) #final dimension 1*nw(1 + ns)*nw

  def cuda(self):
    self.encoder.cuda()
    self.fc.cuda()
    self.gnn.cuda()
    self.support_label = self.support_label.cuda()
    return self

  def set_forward(self,x,is_feature=False,nq_feature=15):
    x = x.cuda()

    if is_feature:
      # reshape the feature tensor: n_way * n_s + 15 * f
      assert(x.size(1) == self.n_support + nq_feature)
      z = self.fc(x.view(-1, *x.size()[2:]))
      z = z.view(self.n_way, -1, z.size(1))
    else:
      # get feature using encoder
      # x has size: n_way*(n_support + n_query) n_channel num_frames img_size img_size
      z = self.encoder(x)
      z = self.avgpool(z).squeeze() #n_way*(n_support + n_query) 512
      z = self.fc(z) # n_way*(n_support + n_query) 128
      z = z.view(self.n_way, -1, z.size(1)) # n_way (n_support + n_query) 128

    # stack the feature for metric function: n_way * n_s + n_q * f -> n_q * [1 * n_way(n_s + 1) * f]
    z_stack = [torch.cat([z[:, :self.n_support], z[:, self.n_support + i:self.n_support + i + 1]], dim=1).view(1, -1, z.size(2)) for i in range(self.n_query)]
    
    assert(z_stack[0].size(1) == self.n_way*(self.n_support + 1))
    scores = self.forward_gnn(z_stack)
    return scores


  def MAML_update(self):
    names = []
    for name, param in self.encoder.named_parameters():
      if param.requires_grad:
        #print(name)
        names.append(name)
    
    names_sub = names[:-12]
    if not self.first:
      for (name, param), (name1, param1), (name2, param2) in zip(self.encoder.named_parameters(), self.encoder2.named_parameters(), self.encoder3.named_parameters()):
        if name not in names_sub:
          dat_change = param2.data - param1.data ### Y - X
          new_dat = param.data - dat_change ### (Y- V) - (Y-X) = X-V
          param.data.copy_(new_dat)

  
  def set_forward_finetune(self,x,is_feature=False):
    # x is the input data with shape: n_way(n_support + n_query) * nc * n_frame * img_size * img_size
    x = x.cuda()
    # get feature using encoder
    batch_size = 4
    support_size = self.n_way * self.n_support 

    for name, param  in self.encoder.named_parameters():
      param.requires_grad = True
    
    x_var = Variable(x)
    # reshape x_var to n_way*(n_support + n_query) * nc * n_frame * img_size * img_size
    x_var = x_var.view(self.n_way,(self.n_support+self.n_query),*x.size()[1:])
      
    y_a_i = Variable( torch.from_numpy( np.repeat(range( self.n_way ), self.n_support ) )).cuda() # (25,)

    #print(y_a_i)
    # after the first time run set_forward_finetune
    # self.first will become false and 
    # self.encoder2 and self.encoder3 will be updated
    # seems self.encoder2 is the previsou feat_net before fine tune 
    # self.encoder3 is the feat_net after fine tune 
    # self.encoder is also the feat_net after fine tune
    self.MAML_update() ## call MAML update
    
    # x_b_i is the query samples
    # x_a_i is the support samples
    x_b_i = x_var[:, self.n_support:,:,:,:].contiguous().view( self.n_way* self.n_query,   *x.size()[1:]) 
    x_a_i = x_var[:,:self.n_support,:,:,:].contiguous().view( self.n_way* self.n_support, *x.size()[1:]) # (25, 3, 32, 112, 112)
    
    feat_network = copy.deepcopy(self.encoder)
    classifier = Classifier(self.feat_dim, self.n_way)
    delta_opt = torch.optim.Adam(filter(lambda p: p.requires_grad, feat_network.parameters()), lr = 0.01)
    loss_fn = nn.CrossEntropyLoss().cuda() ##change this code up ## dorop n way
    classifier_opt = torch.optim.Adam(classifier.parameters(), lr = 0.01, weight_decay=0.001) ##try it with weight_decay
    
    names = []
    for name, param in feat_network.named_parameters():
      if param.requires_grad:
        #print(name)
        names.append(name)
    
    names_sub = names[:-12] ### last R2plus1d block can adapt

    for name, param in feat_network.named_parameters():
      if name in names_sub:
        param.requires_grad = False    
  
      
    total_epoch = 15

    classifier.train()
    feat_network.train()

    classifier.cuda()
    feat_network.cuda()


    for epoch in range(total_epoch):
          rand_id = np.random.permutation(support_size)

          for j in range(0, support_size, batch_size):
              classifier_opt.zero_grad()
              
              delta_opt.zero_grad()

              #####################################
              selected_id = torch.from_numpy( rand_id[j: min(j+batch_size, support_size)]).cuda()
              #discard the last batch if cannot be filled
              if len(selected_id) < batch_size:
                continue
              z_batch = x_a_i[selected_id]
              y_batch = y_a_i[selected_id] 
              #####################################

              output = feat_network(z_batch)
              #ZL added
              output = self.avgpool(output).squeeze()
              logits = classifier(output)

              loss = loss_fn(logits, y_batch)
              #grad = torch.autograd.grad(set_loss, fast_parameters, create_graph=True)

              #####################################
              loss.backward() ### think about how to compute gradients and achieve a good initialization

              classifier_opt.step()
              delta_opt.step()
    

    #feat_network.eval() ## fix this
    #classifier.eval()
    #self.train() ## continue training this!
    if self.first == True:
      self.first = False
    self.encoder2 = copy.deepcopy(self.encoder)
    self.encoder3 = copy.deepcopy(feat_network) ## before the new state_dict is copied over
    self.encoder.load_state_dict(feat_network.state_dict())
    
    for name, param  in self.encoder.named_parameters():
        param.requires_grad = True
    
    output_support = self.encoder(x_a_i.cuda())
    output_support = self.avgpool(output_support).squeeze()
    output_support = output_support.view(self.n_way, self.n_support, -1)

    output_query = self.encoder(x_b_i.cuda())
    output_query = self.avgpool(output_query).squeeze()
    output_query = output_query.view(self.n_way,self.n_query,-1)

    final = torch.cat((output_support, output_query), dim =1).cuda()
    #print(x.size(1))
    #print(x.shape)
    # assert(final.size(1) == self.n_support + 16) ##16 query samples in each batch
    z = self.fc(final.view(-1, *final.size()[2:]))
    z = z.view(self.n_way, -1, z.size(1))

    z_stack = [torch.cat([z[:, :self.n_support], z[:, self.n_support + i:self.n_support + i + 1]], dim=1).view(1, -1, z.size(2)) for i in range(self.n_query)]
    
    assert(z_stack[0].size(1) == self.n_way*(self.n_support + 1))
    
    scores = self.forward_gnn(z_stack)
    
    return scores

  #mamal update requires too much memory 
  #so if the memry is limited use this versiion of fitune without mamal update
  def set_forward_finetune_without_mamal(self,x,is_feature=False):
    # x is the input data with shape: n_way(n_support + n_query) * nc * n_frame * img_size * img_size
    x = x.cuda()
    
    # get feature using encoder
    batch_size = 4
    support_size = self.n_way * self.n_support 

    for name, param  in self.encoder.named_parameters():
      param.requires_grad = True
    
    x_var = Variable(x)
    # reshape x_var to n_way*(n_support + n_query) * nc * n_frame * img_size * img_size
    x_var = x_var.view(self.n_way,(self.n_support+self.n_query),*x.size()[1:])
    
    y_a_i = Variable( torch.from_numpy( np.repeat(range( self.n_way ), self.n_support ) )).cuda() # (25,)

    # x_b_i is the query samples
    # x_a_i is the support samples
    x_b_i = x_var[:, self.n_support:,:,:,:].contiguous().view( self.n_way* self.n_query,   *x.size()[1:]) 
    x_a_i = x_var[:,:self.n_support,:,:,:].contiguous().view( self.n_way* self.n_support, *x.size()[1:]) # (25, 3, 32, 112, 112)
    

    classifier = Classifier(self.feat_dim, self.n_way)
    delta_opt = torch.optim.Adam(filter(lambda p: p.requires_grad, self.encoder.parameters()), lr = 0.01)
    loss_fn = nn.CrossEntropyLoss().cuda() ##change this code up ## dorop n way
    classifier_opt = torch.optim.Adam(classifier.parameters(), lr = 0.01, weight_decay=0.001) ##try it with weight_decay
    
    names = []
    for name, param in self.encoder.named_parameters():
      if param.requires_grad:
        #print(name)
        names.append(name)
    
    names_sub = names[:-12] ### last R2plus1d block can adapt

    for name, param in self.encoder.named_parameters():
      if name in names_sub:
        param.requires_grad = False    
  
      
    total_epoch = 15

    classifier.train()
    self.encoder.train()

    classifier.cuda()
    #self.encoder should be on cuda already

    for epoch in range(total_epoch):
          rand_id = np.random.permutation(support_size)
          for j in range(0, support_size, batch_size):
              classifier_opt.zero_grad()
              delta_opt.zero_grad()

              #####################################
              selected_id = torch.from_numpy( rand_id[j: min(j+batch_size, support_size)]).cuda()
              #discard the last batch if cannot be filled
              if len(selected_id) < batch_size:
                continue
              z_batch = x_a_i[selected_id]
              y_batch = y_a_i[selected_id] 
              #####################################

              output = self.encoder(z_batch)
              #ZL added
              output = self.avgpool(output).squeeze()
              logits = classifier(output)

              loss = loss_fn(logits, y_batch)
              #grad = torch.autograd.grad(set_loss, fast_parameters, create_graph=True)

              #####################################
              loss.backward() ### think about how to compute gradients and achieve a good initialization

              classifier_opt.step()
              delta_opt.step()
    
    for name, param  in self.encoder.named_parameters():
        param.requires_grad = True
    
    output_support = self.encoder(x_a_i.cuda())
    output_support = self.avgpool(output_support).squeeze()
    output_support = output_support.view(self.n_way, self.n_support, -1)

    output_query = self.encoder(x_b_i.cuda())
    output_query = self.avgpool(output_query).squeeze()
    output_query = output_query.view(self.n_way,self.n_query,-1)

    final = torch.cat((output_support, output_query), dim =1).cuda()
    #print(x.size(1))
    #print(x.shape)
    # assert(final.size(1) == self.n_support + 16) ##16 query samples in each batch
    z = self.fc(final.view(-1, *final.size()[2:]))
    z = z.view(self.n_way, -1, z.size(1))

    z_stack = [torch.cat([z[:, :self.n_support], z[:, self.n_support + i:self.n_support + i + 1]], dim=1).view(1, -1, z.size(2)) for i in range(self.n_query)]
    
    assert(z_stack[0].size(1) == self.n_way*(self.n_support + 1))
    
    scores = self.forward_gnn(z_stack)
    
    return scores

  def forward_gnn(self, zs):
    # note here actually n_q is 1
    # gnn inp: n_q * n_way(n_s + 1) * (f + n_way)
    # z dim is zs: 1*n_way(n_s + 1)*f
    # f is feature size
    # support_label dim: 1*n_way(n_s + 1)*n_way
    nodes = torch.cat([torch.cat([z, self.support_label], dim=2) for z in zs], dim=0)
    scores = self.gnn(nodes) # score dim: nq*n_way(n_s + 1)*n_way

    # n_q * n_way(n_s + 1) * n_way -> (n_way * n_q) * n_way
    # the last one on dim 2 is the socre of the query sample again actually the n_query is 1 
    # didn't quite see the necessary of the permute operation
    scores = scores.view(self.n_query, self.n_way, self.n_support + 1, self.n_way)[:, :, -1].permute(1, 0, 2).contiguous().view(-1, self.n_way)
    return scores

  # def set_forward_loss(self, x):
  #   # y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query))
  #   # y_query = y_query.cuda()
  #   scores = self.set_forward(x)
  #   return scores

  # def set_forward_loss_finetune(self, x):
  #   # y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query))
  #   # y_query = y_query.cuda()
  #   scores = self.set_forward_finetune(x)
  #   return scores

  # def set_forward_loss_finetune_without_maml(self, x):
  #   # y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query))
  #   # y_query = y_query.cuda()
  #   scores = self.set_forward_finetune_without_mamal(x)
  #   return scores