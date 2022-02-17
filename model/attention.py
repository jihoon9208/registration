#from _typeshed import Self
import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import gc
from tools.utils import square_distance, index_points
import MinkowskiEngine as ME

    

    
def farthest_point_sample(xyz, npoint):
    """
    get_graph_feature
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape #N 점의 갯수, C 차원
    
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)

    
    for i in range(npoint):
        centroids[:,i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1, dtype=torch.float32)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    
    del farthest ,mask, dist ,distance
    del batch_indices

    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    
    group_idx[sqrdists > radius ** 2] = N
    del sqrdists
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]

    del group_first, mask

    return group_idx

def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, C]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, 1, N, C+D]
    """
    B, N, C = xyz.shape
    S = npoint

    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        fps_points = index_points(points, fps_idx)
        del fps_idx, idx
        fps_points = torch.cat([new_xyz, fps_points], dim=-1)
        #new_points = grouped_points
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
        fps_points = new_xyz
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_points
    else:

        return new_xyz, new_points



class GraphAttention(ME.MinkowskiNetwork):
    def __init__(self, 
                in_channel, 
                feature_dim,
                dropout, 
                alpha,
                D=3):
        ME.MinkowskiNetwork.__init__(self, D)
        
        self.alpha = alpha
        self.a = nn.Parameter(torch.zeros(size=(in_channel, feature_dim)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.in1 = nn.InstanceNorm2d(feature_dim)
        self.dropout = dropout
        self.in_channel = in_channel
        self.leakyrelu = nn.LeakyReLU(self.alpha)
    
    def forward(self, center_xyz, center_feature, grouped_xyz, grouped_feature):
        B, npoint, C = center_xyz.size()
        _, _, nsample, D = grouped_feature.size()
        
        delta_p = center_xyz.view(B, npoint, 1, C).expand(B, npoint, nsample, C) - grouped_xyz # [B, npoint, nsample, C]
        delta_h = center_feature.view(B, npoint, 1, D).expand(B, npoint, nsample, D) - grouped_feature # [B, npoint, nsample, D]
        delta_p_concat_h = torch.cat([delta_p,delta_h],dim = -1) # [B, npoint, nsample, C+D]
        
        t = torch.matmul(delta_p_concat_h, self.a)
        e = self.leakyrelu(t)
        # [B, npoint, nsample,D]
        # attention distribution
        attention = F.softmax(e, dim=2) # [B, npoint, nsample,D]
        attention = F.dropout(attention, self.dropout, training=self.training)
        graph_pooling = torch.sum(torch.mul(attention, grouped_feature),dim = 2) # [B, npoint, D]
       
        return graph_pooling


class GraphAttentionConvLayer(ME.MinkowskiNetwork):
    def __init__(self, npoint, radius, nsample, in_channels, mlp, droupout=0.6, alpha=0.2, D=3):
        ME.MinkowskiNetwork.__init__(self, D)
        """
        process self-attention and cross-attention
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]

        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.droupout = droupout

        self.alpha = alpha
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        last_channel = in_channels
        
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        
        self.sGAT = GraphAttention(last_channel+3, last_channel, self.droupout, self.alpha)
    
    def forward(self, xyz, point):
        

        xyz = xyz.permute(0, 2, 1)
        if point is not None:
            point = point.permute(0, 2, 1)
        
        #Sample and group
        with torch.no_grad():
            new_xyz, new_points, grouped_xyz, fps_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, point, True)

        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        fps_points = fps_points.unsqueeze(3).permute(0, 2, 3, 1) # [B, C+D, 1,npoint]

        #Self_Attention
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            fps_points = F.relu(bn(conv(fps_points)))
            new_points =  F.relu(bn(conv(new_points))) 

           

        new_points = self.sGAT(center_xyz=new_xyz,
                              center_feature=fps_points.squeeze(dim=2).permute(0,2,1),
                              grouped_xyz=grouped_xyz,
                              grouped_feature=new_points.permute(0,3,2,1))
        
        new_xyz = new_xyz.permute(0, 2, 1)
        new_points = new_points.permute(0, 2, 1)

        gc.collect()
        torch.cuda.empty_cache()
        
        return new_xyz, new_points #[C, nsample] [D, nsample]

def MLP(channels: list, do_bn=True):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n-1):
            if do_bn:
                layers.append(nn.InstanceNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)

def attention(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob


class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """
    def __init__(self, num_heads, d_model):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value):
        batch_dim = query.size(0)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, _ = attention(query, key, value)
        return self.merge(x.contiguous().view(batch_dim, self.dim*self.num_heads, -1))

class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim, num_heads):
        super().__init__()
        self.att = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim*2, feature_dim*2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)


    def forward(self, x, source):
        message = self.att(x, source, source)
        return self.mlp(torch.cat([x, message], dim = 1 ))


class PointNetFeaturePropagation(ME.MinkowskiNetwork):
    def __init__(self, in_channel, mlp, D=3):
        ME.MinkowskiNetwork.__init__(self, D)
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self,xyz1, xyz2, points1, points2 ):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        
        points2 = points2.permute(0, 2, 1)  #[B,N,D]

        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]
            dists[dists < 1e-10] = 1e-10
            weight = 1.0 / dists  # [B, N, 3]
            weight = weight / torch.sum(weight, dim=-1).view(B, N, 1)  # [B, N, 3]
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)

        else:
            new_points = interpolated_points

        del interpolated_points, dists, idx, weight

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))

        #new_points = new_points.squeeze(dim = 0)

        return new_points

class GCN(ME.MinkowskiNetwork):
    def __init__(self):
        ME.MinkowskiNetwork.__init__(self, D=3)

        self.SelfAttention = GraphAttentionConvLayer(128, 0.8, 32, 256 + 3, [256, 256, 512], D=3)
        self.CrossAttention = AttentionalPropagation(512, 4)
        self.Interpolation = PointNetFeaturePropagation(768, [256, 256])

    def forward(self, xyz1, xyz2, points1, points2):
        
        new_xyz1, new_points1 = self.SelfAttention(xyz1, points1)
        new_xyz2, new_points2 = self.SelfAttention(xyz2, points2)

        new_points1 = new_points1 + self.CrossAttention(new_points1, new_points2)
        new_points2 = new_points2 + self.CrossAttention(new_points2, new_points1)

        new_points1 = self.Interpolation(xyz1, new_xyz1, points1, new_points1)
        new_points2 = self.Interpolation(xyz2, new_xyz2, points2, new_points2)
        
        return new_points1, new_points2