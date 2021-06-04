import torch
from torch.autograd import Variable
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.weight_norm import WeightNorm

def init_layer(L):
    # Initialization using fan-in
    if isinstance(L, nn.Conv2d):
        n = L.kernel_size[0]*L.kernel_size[1]*L.out_channels
        L.weight.data.normal_(0,math.sqrt(2.0/float(n)))
    elif isinstance(L, nn.BatchNorm2d):
        L.weight.data.fill_(1)
        L.bias.data.fill_(0)

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

# --- Linear module ---
class Linear_fw(nn.Linear): #used in MAML to forward input with fast weight
  def __init__(self, in_features, out_features, bias=True):
    super(Linear_fw, self).__init__(in_features, out_features, bias=bias)
    self.weight.fast = None #Lazy hack to add fast weight link
    self.bias.fast = None

  def forward(self, x):
    if self.weight.fast is not None and self.bias.fast is not None:
      out = F.linear(x, self.weight.fast, self.bias.fast)
    else:
      out = super(Linear_fw, self).forward(x)
    return out

# --- Conv2d module ---
class Conv2d_fw(nn.Conv2d): #used in MAML to forward input with fast weight
  def __init__(self, in_channels, out_channels, kernel_size, stride=1,padding=0, bias = True):
    super(Conv2d_fw, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
    self.weight.fast = None
    if not self.bias is None:
      self.bias.fast = None

  def forward(self, x):
    if self.bias is None:
      if self.weight.fast is not None:
        out = F.conv2d(x, self.weight.fast, None, stride= self.stride, padding=self.padding)
      else:
        out = super(Conv2d_fw, self).forward(x)
    else:
      if self.weight.fast is not None and self.bias.fast is not None:
        out = F.conv2d(x, self.weight.fast, self.bias.fast, stride= self.stride, padding=self.padding)
      else:
        out = super(Conv2d_fw, self).forward(x)
    return out

# --- BatchNorm1d ---
class BatchNorm1d_fw(nn.BatchNorm1d):
  def __init__(self, num_features, momentum=0.1, track_running_stats=True):
    super(BatchNorm1d_fw, self).__init__(num_features, momentum=momentum, track_running_stats=track_running_stats)
    self.weight.fast = None
    self.bias.fast = None
    if self.track_running_stats:
      self.register_buffer('running_mean', torch.zeros(num_features))
      self.register_buffer('running_var', torch.zeros(num_features))
    self.reset_parameters()

  def reset_running_stats(self):
    if self.track_running_stats:
      self.running_mean.zero_()
      self.running_var.fill_(1)

  def forward(self, x, step=0):
    if self.weight.fast is not None and self.bias.fast is not None:
      weight = self.weight.fast
      bias = self.bias.fast
    else:
      weight = self.weight
      bias = self.bias
    if self.track_running_stats:
      out = F.batch_norm(x, self.running_mean, self.running_var, weight, bias, training=self.training, momentum=self.momentum)
    else:
      out = F.batch_norm(x, torch.zeros(x.size(1), dtype=x.dtype, device=x.device), torch.ones(x.size(1), dtype=x.dtype, device=x.device), weight, bias, training=True, momentum=1)
    return out


# --- Conv2d module ---
class Conv2d_fw(nn.Conv2d): #used in MAML to forward input with fast weight
  def __init__(self, in_channels, out_channels, kernel_size, stride=1,padding=0, bias = True):
    super(Conv2d_fw, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
    self.weight.fast = None
    if not self.bias is None:
      self.bias.fast = None

  def forward(self, x):
    if self.bias is None:
      if self.weight.fast is not None:
        out = F.conv2d(x, self.weight.fast, None, stride= self.stride, padding=self.padding)
      else:
        out = super(Conv2d_fw, self).forward(x)
    else:
      if self.weight.fast is not None and self.bias.fast is not None:
        out = F.conv2d(x, self.weight.fast, self.bias.fast, stride= self.stride, padding=self.padding)
      else:
        out = super(Conv2d_fw, self).forward(x)
    return out

  # --- softplus module ---
def softplus(x):
  return torch.nn.functional.softplus(x, beta=100)

# --- BatchNorm2d ---
class BatchNorm2d_fw(nn.BatchNorm2d):
  def __init__(self, num_features, momentum=0.1, track_running_stats=True):
    super(BatchNorm2d_fw, self).__init__(num_features, momentum=momentum, track_running_stats=track_running_stats)
    self.weight.fast = None
    self.bias.fast = None
    if self.track_running_stats:
      self.register_buffer('running_mean', torch.zeros(num_features))
      self.register_buffer('running_var', torch.zeros(num_features))
    self.reset_parameters()

  def reset_running_stats(self):
    if self.track_running_stats:
      self.running_mean.zero_()
      self.running_var.fill_(1)

  def forward(self, x, step=0):
    if self.weight.fast is not None and self.bias.fast is not None:
      weight = self.weight.fast
      bias = self.bias.fast
    else:
      weight = self.weight
      bias = self.bias
    if self.track_running_stats:
      out = F.batch_norm(x, self.running_mean, self.running_var, weight, bias, training=self.training, momentum=self.momentum)
    else:
      out = F.batch_norm(x, torch.zeros(x.size(1), dtype=x.dtype, device=x.device), torch.ones(x.size(1), dtype=x.dtype, device=x.device), weight, bias, training=True, momentum=1)
    return out

# --- BatchNorm1d ---
class BatchNorm1d_fw(nn.BatchNorm1d):
  def __init__(self, num_features, momentum=0.1, track_running_stats=True):
    super(BatchNorm1d_fw, self).__init__(num_features, momentum=momentum, track_running_stats=track_running_stats)
    self.weight.fast = None
    self.bias.fast = None
    if self.track_running_stats:
      self.register_buffer('running_mean', torch.zeros(num_features))
      self.register_buffer('running_var', torch.zeros(num_features))
    self.reset_parameters()

  def reset_running_stats(self):
    if self.track_running_stats:
      self.running_mean.zero_()
      self.running_var.fill_(1)

  def forward(self, x, step=0):
    if self.weight.fast is not None and self.bias.fast is not None:
      weight = self.weight.fast
      bias = self.bias.fast
    else:
      weight = self.weight
      bias = self.bias
    if self.track_running_stats:
      out = F.batch_norm(x, self.running_mean, self.running_var, weight, bias, training=self.training, momentum=self.momentum)
    else:
      out = F.batch_norm(x, torch.zeros(x.size(1), dtype=x.dtype, device=x.device), torch.ones(x.size(1), dtype=x.dtype, device=x.device), weight, bias, training=True, momentum=1)
    return out

# --- feature-wise transformation layer ---
class FeatureWiseTransformation2d_fw(nn.BatchNorm2d):
  def __init__(self, num_features, momentum=0.1, track_running_stats=True):
    super(FeatureWiseTransformation2d_fw, self).__init__(num_features, momentum=momentum, track_running_stats=track_running_stats)
    self.weight.fast = None
    self.bias.fast = None
    if self.track_running_stats:
      self.register_buffer('running_mean', torch.zeros(num_features))
      self.register_buffer('running_var', torch.zeros(num_features))
    self.gamma = torch.nn.Parameter(torch.ones(1, num_features, 1, 1)*0.3)
    self.beta  = torch.nn.Parameter(torch.ones(1, num_features, 1, 1)*0.5)
    self.gamma.requires_grad = False
    self.beta.requires_grad = False
    self.reset_parameters()

  def reset_running_stats(self):
    if self.track_running_stats:
      self.running_mean.zero_()
      self.running_var.fill_(1)

  def forward(self, x, step=0):
    if self.weight.fast is not None and self.bias.fast is not None:
      weight = self.weight.fast
      bias = self.bias.fast
    else:
      weight = self.weight
      bias = self.bias
    if self.track_running_stats:
      out = F.batch_norm(x, self.running_mean, self.running_var, weight, bias, training=self.training, momentum=self.momentum)
    else:
      out = F.batch_norm(x, torch.zeros_like(x), torch.ones_like(x), weight, bias, training=True, momentum=1)

    # apply feature-wise transformation
    if self.training:
      gamma = (1 + torch.randn(1, self.num_features, 1, 1, dtype=self.gamma.dtype, device=self.gamma.device)*softplus(self.gamma)).expand_as(out)
      beta = (torch.randn(1, self.num_features, 1, 1, dtype=self.beta.dtype, device=self.beta.device)*softplus(self.beta)).expand_as(out)
      out = gamma*out + beta
    return out






