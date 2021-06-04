import numpy as np
import torch
import torch.nn as nn

# dynamic graph from knn
def knn(x, y=None, k=10):
    """
    :param x: BxCxN
    :param y: BxCxM
    :param k: scalar
    :return: BxMxk
    """
    if y is None:
        y = x
    # logging.info('Size in KNN: {} - {}'.format(x.size(), y.size()))
    inner = -2 * torch.matmul(y.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    yy = torch.sum(y ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - yy.transpose(2, 1)
    # if k > pairwise_distance.shape[-1]: k = pairwise_distance.shape[-1]
    _, idx = pairwise_distance.topk(k=k, dim=-1)  # (batch_size, num_points, k)
    return idx


# get graph feature
def get_graph_feature(x, prev_x=None, k=20, idx_knn=None, r=-1, style=0):
    """
    :param x:
    :param prev_x:
    :param k:
    :param idx:
    :param r: output downsampling factor (-1 for no downsampling)
    :param style: method to get graph feature
    :return:
    """
    batch_size = x.size(0)
    num_points = x.size(2)  # if prev_x is None else prev_x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx_knn is None:
        idx_knn = knn(x=x, y=prev_x, k=k)  # (batch_size, num_points, k)
    else:
        k = idx_knn.shape[-1]
    # print(idx_knn.shape)
    device = x.device  # torch.device('cuda')
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx_knn + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    if style == 0:  # use offset as feature
        feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2)
    elif style == 1:  # use feature as feature
        feature = torch.cat((feature, x), dim=3).permute(0, 3, 1, 2)
    else: # style == 2:
        feature = feature.permute(0,3,1,2)
    # downsample if needed
    if r != -1:
        select_idx = torch.from_numpy(np.random.choice(feature.size(2), feature.size(2) // r,
                                                       replace=False)).to(device=device)
        feature = feature[:, :, select_idx, :]
    return feature, idx_knn


# basic block
class GCNeXt(nn.Module):
    def __init__(self, 
                 channel_in, 
                 channel_out, 
                 k=3, 
                 norm_layer=None, 
                 groups=32, 
                 width_group=4,
                 enable_scale=False
                 ):
        super(GCNeXt, self).__init__()
        self.k = k
        self.groups = groups
        self.enable_scale = enable_scale

        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        width = width_group * groups
        self.tconvs = nn.Sequential(
            nn.Conv1d(channel_in, width, kernel_size=1), nn.ReLU(True),
            nn.Conv1d(width, width, kernel_size=3, groups=groups, padding=1), nn.ReLU(True),
            nn.Conv1d(width, channel_out, kernel_size=1),
        ) # temporal graph

        self.sconvs = nn.Sequential(
            nn.Conv2d(channel_in * 2, width, kernel_size=1), nn.ReLU(True),
            nn.Conv2d(width, width, kernel_size=1, groups=groups), nn.ReLU(True),
            nn.Conv2d(width, channel_out, kernel_size=1),
        ) # semantic graph

        self.relu = nn.ReLU(True)
        self.sacle = nn.Parameter(torch.FloatTensor([1.0, 1.0, 1.0]))

    def forward(self, x):
        # note here 100 is the number of the nodes
        # x dim: (bs,ch,100)
        # tout dim: (bs,ch,100)
        identity = x  # residual
        tout = self.tconvs(x)  # conv on temporal graph
        # if the num of neighbors is greater than the num of nodes then select all the nodes
        k = min(self.k, x.size()[-1])
        x_f, idx = get_graph_feature(x, k=k, style=1)  # (bs,ch,100) -> (bs, 2ch, 100, k)
        sout = self.sconvs(x_f)  # conv on semantic graph
        sout = sout.max(dim=-1, keepdim=False)[0]  # (bs, ch, 100, k) -> (bs, ch, 100)
        if self.enable_scale:
            out = self.sacle[0] * identity + self.sacle[1] * tout + self.sacle[2] * sout
        else:
            out = tout + identity + sout  # fusion

        return self.relu(out)


class STGCN(nn.Module):
    def __init__(self, n_way, 
    num_neighbors=3, 
    feat_dim=512, 
    h_dim_1d=128, 
    gcn_groups=32, 
    bk2=False,
    fuse_type="mean",
    enable_gcn_scale=False):
        super(STGCN, self).__init__()
        self.feat_dim = feat_dim
        self.h_dim_1d = h_dim_1d
        self.n_way = n_way
        self.bk2 = bk2
        self.k = num_neighbors
        self.gcn_groups = gcn_groups
        self.fuse_type = fuse_type
        self.enable_gcn_scale = enable_gcn_scale
        # Backbone Part 1
        self.backbone1 = nn.Sequential(
            nn.Conv1d(self.feat_dim, self.h_dim_1d, kernel_size=3, padding=1, groups=4),
            nn.ReLU(inplace=True),
            GCNeXt(self.h_dim_1d, self.h_dim_1d, 
            k=self.k, groups=self.gcn_groups, 
            enable_scale=self.enable_gcn_scale),
        )
        if self.bk2:
            # Backbone Part 2
            self.backbone2 = nn.Sequential(
                GCNeXt(self.h_dim_1d, self.h_dim_1d, k=self.k, 
                groups=self.gcn_groups, enable_scale=self.enable_gcn_scale),
            )

        self.last_layer= nn.Linear(self.h_dim_1d, self.n_way)

    def forward(self, snip_feature):
        # snip_feature dim: (bs, ch, num_points)
        ctx_feature = self.backbone1(snip_feature).contiguous()  # (bs, 2048, 256) -> (bs, 256, 256)
        if self.bk2:
            ctx_feature = self.backbone2(ctx_feature)  #
        # average the node features to get the final representation of the graph
        if self.fuse_type == "mean":
            feature = torch.mean(ctx_feature, dim=-1)
        elif self.fuse_type == "max":
            #note torch.max return (value, indices)
            feature = torch.max(ctx_feature, dim=-1)[0]
        scores = self.last_layer(feature)
        return scores

    def extract_feature(self, snip_feature):
        # snip_feature dim: (bs, ch, num_points)
        ctx_feature = self.backbone1(snip_feature).contiguous()  # (bs, 2048, 256) -> (bs, 256, 256)
        if self.bk2:
            ctx_feature = self.backbone2(ctx_feature)  #
        # average the node features to get the final representation of the graph
        if self.fuse_type == "mean":
            feature = torch.mean(ctx_feature, dim=-1)
        elif self.fuse_type == "max":
            #note torch.max return (value, indices)
            feature = torch.max(ctx_feature, dim=-1)[0]
        return feature
