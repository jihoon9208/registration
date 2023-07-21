
import torch
import torch.nn.functional as F
import torch.nn as nn

from copy import deepcopy
import torch.utils.checkpoint as checkpoint

from tools.utils import square_distance_tmp

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


class SelfAttention(nn.Module):
    def __init__(self,feature_dim,k=10):
        super(SelfAttention, self).__init__() 
        self.conv1 = nn.Conv2d(feature_dim, feature_dim * 2 , kernel_size=1, bias=False)
        self.in1 = nn.InstanceNorm2d(feature_dim)
        
        self.conv2 = nn.Conv2d(feature_dim , feature_dim * 2, kernel_size=1,bias=False)
        self.in2 = nn.InstanceNorm2d(feature_dim)

        self.conv3 = nn.Conv2d(feature_dim , feature_dim * 2, kernel_size=1,bias=False)
        self.in3 = nn.InstanceNorm2d(feature_dim)

        self.conv4 = nn.Conv2d(feature_dim * 2, feature_dim, kernel_size=1, bias=False)
        self.in4 = nn.InstanceNorm2d(feature_dim)


        self.k = k

    def forward(self, feat):
        
        # Here we take coordinats and features, feature aggregation are guided by coordinates
        # Input: 
        #     coords:     [B, 3, N]
        #     feats:      [B, C, N]
        # Output:
        #     feats:      [B, C, N]
        

        B, C, N = feat.size()

        x0 = feat.unsqueeze(-1)  #[B, C, N, 1]

        #x1 = get_graph_feature(coord, x0.squeeze(-1), self.k)
        x1 = F.leaky_relu(self.in1(self.conv1(x0)), negative_slope=0.2)
        x1 = x1.max(dim=-1,keepdim=True)[0]

        #x2 = get_graph_feature(coord, x1.squeeze(-1), self.k)
        x2 = F.leaky_relu(self.in2(self.conv2(x0)), negative_slope=0.2)
        x2 = x2.max(dim=-1, keepdim=True)[0]

        x3 = F.leaky_relu(self.in3(self.conv3(x0)), negative_slope=0.2)
        x3 = x3.max(dim=-1, keepdim=True)[0]

        query_key = torch.matmul(x1.squeeze(-1).permute(0, 2, 1) , x2.squeeze(-1))

        x4 = torch.matmul(x3.squeeze(-1), query_key)

        x4 = x1.squeeze(-1) - x4

        x4 = F.leaky_relu(self.in4(self.conv4(x4.unsqueeze(-1))), negative_slope=0.2)
        x4 = x4.max(dim=-1, keepdim=True)[0]

        return x4.squeeze(-1)


