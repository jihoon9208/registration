
import torch
import torch.nn.functional as F
import torch.nn as nn
import MinkowskiEngine as ME

from copy import deepcopy
import torch.utils.checkpoint as checkpoint

from tools.utils import square_distance, square_distance_tmp

def get_graph_feature(coords, feats, k=10):
    """
    Apply KNN search based on coordinates, then concatenate the features to the centroid features
    Input:
        X:          [B, 3, N]
        feats:      [B, C, N]
    Return:
        feats_cat:  [B, 2C, N, k]
    """
    # apply KNN search to build neighborhood
    B, C, N = feats.size()
    dist = square_distance_tmp(coords.transpose(1,2), coords.transpose(1,2))

    idx = dist.topk(k=k+1, dim=-1, largest=False, sorted=True)[1]  #[B, N, K+1], here we ignore the smallest element as it's the query itself  
    idx = idx[:,:,1:]  #[B, N, K]

    idx = idx.unsqueeze(1).repeat(1,C,1,1) #[B, C, N, K]
    all_feats = feats.unsqueeze(2).repeat(1, 1, N, 1)  # [B, C, N, N]

    neighbor_feats = torch.gather(all_feats, dim=-1,index=idx) #[B, C, N, K]

    # concatenate the features with centroid
    feats = feats.unsqueeze(-1).repeat(1,1,1,k)

    feats_cat = torch.cat((feats, neighbor_feats-feats),dim=1)

    return feats_cat

""" 
class SelfAttention(nn.Module):
    def __init__(self,feature_dim,k=10):
        super(SelfAttention, self).__init__() 
        self.conv1 = nn.Conv2d(feature_dim, feature_dim * 2 , kernel_size=1, bias=False)
        self.in1 = nn.InstanceNorm2d(feature_dim)
        
        self.conv2 = nn.Conv2d(feature_dim * 2, feature_dim * 4, kernel_size=1,bias=False)
        self.in2 = nn.InstanceNorm2d(feature_dim * 4)

        self.conv3 = nn.Conv2d(feature_dim * 7, feature_dim, kernel_size=1, bias=False)
        self.in3 = nn.InstanceNorm2d(feature_dim)

        self.k = k

    def forward(self, coord, feat):
        
        Here we take coordinats and features, feature aggregation are guided by coordinates
        Input: 
            coords:     [B, 3, N]
            feats:      [B, C, N]
        Output:
            feats:      [B, C, N]
        

        B, C, N = feat.size()

        x0 = feat.unsqueeze(-1)  #[B, C, N, 1]

        #x1 = get_graph_feature(coord, x0.squeeze(-1), self.k)
        x1 = F.leaky_relu(self.in1(self.conv1(x0)), negative_slope=0.2)
        x1 = x1.max(dim=-1,keepdim=True)[0]

        #x2 = get_graph_feature(coord, x1.squeeze(-1), self.k)
        x2 = F.leaky_relu(self.in2(self.conv2(x1)), negative_slope=0.2)
        x2 = x2.max(dim=-1, keepdim=True)[0]

        x3 = torch.cat((x0,x1,x2),dim=1)
        x3 = F.leaky_relu(self.in3(self.conv3(x3)), negative_slope=0.2).view(B, -1, N)

        x3 = x3.permute(0, 2, 1).squeeze(0)
        coord = coord.permute(0, 2, 1).squeeze(0)
        
        #x3 = x3 + feat
        return x3

"""

class SelfAttention(ME.MinkowskiModule):
    
    def __init__(self, in_channels, out_channels):
        super(SelfAttention, self).__init__()

        self.query = ME.MinkowskiLinear(in_channels, out_channels)
        self.key = ME.MinkowskiLinear(in_channels, out_channels)
        self.value = ME.MinkowskiLinear(in_channels, out_channels)

    def forward(self, x):

        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        attention = torch.matmul(query, key.transpose(-2, -1))
        attention = torch.softmax(attention, dim=-1)
        out = torch.matmul(attention, value)

        return out

