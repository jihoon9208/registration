import sys
import torch
import numpy as np
from typing import Any
import torch.nn as nn
import torch.nn.functional as F

import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MEF

from tools.model_util import soft_BBS_loss_torch, guess_best_alpha_torch, cdist_torch
from tools.radius import compute_graph_nn
from model.common import get_norm
from model.residual_block import get_block
from model.attention import GCN

sys.path.append("./lib/model_exc")
sys.path.append("./lib/model_exc")

import libply_c

#from torch_scatter import scatter_add, scatter_mean


def point_permute(p):
    return p.permute(1,0)
  
def xyz_permute(xyz):
    return xyz.permute(1,0)

def det_3x3(self, mat):

    a, b, c = mat[:, 0, 0], mat[:, 0, 1], mat[:, 0, 2]
    d, e, f = mat[:, 1, 0], mat[:, 1, 1], mat[:, 1, 2]
    g, h, i = mat[:, 2, 0], mat[:, 2, 1], mat[:, 2, 2]

    det = a * e * i + b * f * g + c * d * h
    det = det - c * e * g - b * d * i - a * f * h

    return det


class SparseAttention(ME.MinkowskiNetwork):

    NORM_TYPE = 'BN'
    BLOCK_NORM_TYPE = 'BN'
    CHANNELS = [None, 32, 64, 128, 256, 512]
    TR_CHANNELS = [None, 32, 64, 64, 128]

    def __init__(
                    self,
                    config,
                    in_channels=3,
                    out_channels=32,
                    bn_momentum=0.1,
                    conv1_kernel_size=None,
                    D=3):
        super().__init__(D=3)


        self.voxel_size = config.voxel_size
        NORM_TYPE = self.NORM_TYPE
        BLOCK_NORM_TYPE = self.BLOCK_NORM_TYPE
        CHANNELS = self.CHANNELS
        TR_CHANNELS = self.TR_CHANNELS

        self.conv1 = ME.MinkowskiConvolution(
        in_channels=in_channels,
        out_channels=CHANNELS[1],
        kernel_size = conv1_kernel_size,
        stride=1,
        dilation=1,
        bias=False,
        dimension=D)
        self.norm1 = get_norm(NORM_TYPE, CHANNELS[1], bn_momentum=bn_momentum, D=D)

        self.block1 = get_block(
            BLOCK_NORM_TYPE, CHANNELS[1], CHANNELS[1], bn_momentum=bn_momentum, D=D)

        self.conv2 = ME.MinkowskiConvolution(
            in_channels=CHANNELS[1],
            out_channels=CHANNELS[2],
            kernel_size=3,
            stride=2,
            dilation=1,
            bias=False,
            dimension=D)
        self.norm2 = get_norm(NORM_TYPE, CHANNELS[2], bn_momentum=bn_momentum, D=D)

        self.block2 = get_block(
            BLOCK_NORM_TYPE, CHANNELS[2], CHANNELS[2], bn_momentum=bn_momentum, D=D)

        self.conv3 = ME.MinkowskiConvolution(
            in_channels=CHANNELS[2],
            out_channels=CHANNELS[3],
            kernel_size=3,
            stride=2,
            dilation=1,
            bias=False,
            dimension=D)
        self.norm3 = get_norm(NORM_TYPE, CHANNELS[3], bn_momentum=bn_momentum, D=D)

        self.block3 = get_block(
            BLOCK_NORM_TYPE, CHANNELS[3], CHANNELS[3], bn_momentum=bn_momentum, D=D)

        self.conv4 = ME.MinkowskiConvolution(
            in_channels=CHANNELS[3],
            out_channels=CHANNELS[4],
            kernel_size=3,
            stride=2,
            dilation=1,
            bias=False,
            dimension=D)
        self.norm4 = get_norm(NORM_TYPE, CHANNELS[4], bn_momentum=bn_momentum, D=D)

        self.block4 = get_block(
            BLOCK_NORM_TYPE, CHANNELS[4], CHANNELS[4], bn_momentum=bn_momentum, D=D)

        self.conv4_tr = ME.MinkowskiConvolutionTranspose(
            in_channels=CHANNELS[4] ,
            out_channels=TR_CHANNELS[4],
            kernel_size=3,
            stride=2,
            dilation=1,
            bias=False,
            dimension=D)
        self.norm4_tr = get_norm(NORM_TYPE, TR_CHANNELS[4], bn_momentum=bn_momentum, D=D)

        self.block4_tr = get_block(
            BLOCK_NORM_TYPE, TR_CHANNELS[4], TR_CHANNELS[4], bn_momentum=bn_momentum, D=D)

        self.conv3_tr = ME.MinkowskiConvolutionTranspose(
            #in_channels=CHANNELS[4] + TR_CHANNELS[4],
            in_channels=CHANNELS[3] + TR_CHANNELS[4],
            out_channels=TR_CHANNELS[3],
            kernel_size=3,
            stride=2,
            dilation=1,
            bias=False,
            dimension=D)
        self.norm3_tr = get_norm(NORM_TYPE, TR_CHANNELS[3], bn_momentum=bn_momentum, D=D)

        self.block3_tr = get_block(
            BLOCK_NORM_TYPE, TR_CHANNELS[3], TR_CHANNELS[3], bn_momentum=bn_momentum, D=D)

        self.conv2_tr = ME.MinkowskiConvolutionTranspose(
            #in_channels=CHANNELS[3] + TR_CHANNELS[3],
            in_channels=CHANNELS[2] + TR_CHANNELS[3],        
            out_channels=TR_CHANNELS[2],
            kernel_size=3,
            stride=2,
            dilation=1,
            bias=False,
            dimension=D)
        self.norm2_tr = get_norm(NORM_TYPE, TR_CHANNELS[2], bn_momentum=bn_momentum, D=D)

        self.block2_tr = get_block(
            BLOCK_NORM_TYPE, TR_CHANNELS[2], TR_CHANNELS[2], bn_momentum=bn_momentum, D=D)

        self.conv1_tr = ME.MinkowskiConvolution(
            in_channels= CHANNELS[1] + TR_CHANNELS[2],
            out_channels=TR_CHANNELS[1],
            kernel_size=1,
            stride=1,
            dilation=1,
            bias=False,
            dimension=D)
        
        ################
        # attention
        self.epsilon = torch.nn.Parameter(torch.tensor(-5.0))
        self.gcn = GCN()
        self.proj_gnn = nn.Conv1d(CHANNELS[4], CHANNELS[4], kernel_size=1, bias=True)
        self.proj_score = nn.Conv1d(CHANNELS[4] ,1,kernel_size=1,bias=True)

        
    def forward(self, source, target):

        # source feature 
        source_s1 = self.conv1(source)
        source_s1 = self.norm1(source_s1)
        source_s1 = self.block1(source_s1)
        s_out = MEF.relu(source_s1)

        source_s2 = self.conv2(s_out)
        source_s2 = self.norm2(source_s2)
        source_s2 = self.block2(source_s2)
        s_out = MEF.relu(source_s2)

        source_s4 = self.conv3(s_out)
        source_s4 = self.norm3(source_s4)
        source_s4 = self.block3(source_s4)
        s_out = MEF.relu(source_s4)

        source_s8 = self.conv4(s_out)
        source_s8 = self.norm4(source_s8)
        source_s8 = self.block4(source_s8)
        s_out = MEF.relu(source_s8)

        # target feature
        target_s1 = self.conv1(target)
        target_s1 = self.norm1(target_s1)
        target_s1 = self.block1(target_s1)
        t_out = MEF.relu(target_s1)

        target_s2 = self.conv2(t_out)
        target_s2 = self.norm2(target_s2)
        target_s2 = self.block2(target_s2)
        t_out = MEF.relu(target_s2)

        target_s4 = self.conv3(t_out)
        target_s4 = self.norm3(target_s4)
        target_s4 = self.block3(target_s4)
        t_out = MEF.relu(target_s4)

        target_s8 = self.conv4(t_out)
        target_s8 = self.norm4(target_s8)
        target_s8 = self.block4(target_s8)
        t_out = MEF.relu(target_s8)

        ####################
        # overlap
        sc_feat = s_out.F.transpose(0,1)[None,:] #[1, D, N]
        ta_feat = t_out.F.transpose(0,1)[None,:] #[1, D, M]

        # [N, 3] [M, 3]
        sc_pcd, ta_pcd = s_out.C[:,1:] * self.voxel_size, t_out.C[:,1:] * self.voxel_size

        # feature Attention get
        sc_feat, ta_feat = self.gcn(sc_pcd.transpose(0,1)[None,:], ta_pcd.transpose(0,1)[None,:], sc_feat, ta_feat)
        sc_feat = F.normalize(sc_feat, p=2, dim=1)[0].transpose(0,1)
        ta_feat = F.normalize(ta_feat, p=2, dim=1)[0].transpose(0,1)

        sc_feat = ME.SparseTensor(sc_feat, 
			coordinate_map_key=s_out.coordinate_map_key,
			coordinate_manager=s_out.coordinate_manager)

        ta_feat = ME.SparseTensor(ta_feat,
			coordinate_map_key=t_out.coordinate_map_key,
			coordinate_manager=t_out.coordinate_manager)
        #####################

        s_out = self.conv4_tr(sc_feat)
        s_out = self.norm4_tr(s_out)
        s_out = self.block4_tr(s_out)
        source_s4_tr = MEF.relu(s_out)

        s_out = ME.cat(source_s4_tr, source_s4)

        s_out = self.conv3_tr(s_out)
        s_out = self.norm3_tr(s_out)
        s_out = self.block3_tr(s_out)
        source_s2_tr = MEF.relu(s_out)

        s_out = ME.cat(source_s2_tr, source_s2)

        s_out = self.conv2_tr(s_out)
        s_out = self.norm2_tr(s_out)
        s_out = self.block2_tr(s_out)
        source_s1_tr = MEF.relu(s_out)

        s_out = ME.cat(source_s1_tr, source_s1)
        s_out = self.conv1_tr(s_out)
        source = MEF.relu(s_out)

        ############################

        t_out = self.conv4_tr(ta_feat)
        t_out = self.norm4_tr(t_out)
        t_out = self.block4_tr(t_out)
        target_s4_tr = MEF.relu(t_out)

        t_out = ME.cat(target_s4_tr, target_s4)

        t_out = self.conv3_tr(t_out)
        t_out = self.norm3_tr(t_out)
        t_out = self.block3_tr(t_out)
        target_s2_tr = MEF.relu(t_out)

        t_out = ME.cat(target_s2_tr, target_s2)

        t_out = self.conv2_tr(t_out)
        t_out = self.norm2_tr(t_out)
        t_out = self.block2_tr(t_out)
        target_s1_tr = MEF.relu(t_out)

        t_out = ME.cat(target_s1_tr, target_s1)
        t_out = self.conv1_tr(t_out)
        target = MEF.relu(t_out)
        
        source = ME.SparseTensor(
            source.F / torch.norm(source.F, p=2, dim=1, keepdim=True),
            coordinate_map_key=source.coordinate_map_key,
            coordinate_manager=source.coordinate_manager)  

        target = ME.SparseTensor(
        target.F / torch.norm(target.F, p=2, dim=1, keepdim=True),
        coordinate_map_key=target.coordinate_map_key,
        coordinate_manager=target.coordinate_manager)

        return source.F, target.F

class FullPointMatcher(ME.MinkowskiNetwork):
    def __init__(self, config):
        self.config = config
        super().__init__(D=3)

        num_feats = 1

        self.sparseAttention = SparseAttention(
            self.config, 
            num_feats,
            config.model_n_out,
            bn_momentum=config.bn_momentum,
            conv1_kernel_size=config.conv1_kernel_size,
            D=3).cuda()

    def forward(self, pcd0, pcd1):

        sa_feat0, sa_feat1 = self.sparseAttention(pcd0, pcd1)
        return sa_feat0, sa_feat1

class DGCNN(nn.Module):
    def __init__(self, emb_dims=512):
        super(DGCNN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=1, bias=False)
        self.conv4 = nn.Conv2d(128, 64, kernel_size=1, bias=False)
        self.conv5 = nn.Conv2d(64, emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(emb_dims)

    def forward(self, x):
        batch_size, num_dims, num_points = x.transpose(2,1).size()
        out = self.conv1(x)
        out = self.bn1(out)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x1 = x.max(dim=-1, keepdim=True)[0]

        x = F.relu(self.bn2(self.conv2(x)))
        x2 = x.max(dim=-1, keepdim=True)[0]

        x = F.relu(self.bn3(self.conv3(x)))
        x3 = x.max(dim=-1, keepdim=True)[0]

        x = F.relu(self.bn4(self.conv4(x)))
        x4 = x.max(dim=-1, keepdim=True)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = F.relu(self.bn5(self.conv5(x))).view(batch_size, -1, num_points)
        return x

class GeoFeatureExtraction(ME.MinkowskiNetwork):
    def __init__(self, config):
        
        super().__init__(D=3)
        self.config = config
        self.k_nn_geof = config.k_nn_geof
        self.voxel_size = config.voxel_size
        self.emb_dims = 32
        self.emb_geof = DGCNN(emb_dims=self.emb_dims)
      

    def forward(self, pcd):
        #
        # PCD N * 3
        # geof N * D
        
        graph_nn = compute_graph_nn(pcd, self.k_nn_geof)
        geof = libply_c.compute_norm(pcd.cpu().numpy(), graph_nn["target"], self.k_nn_geof).astype('float32')
        geof = torch.from_numpy(geof).detach() 
        geof_mlp = self.emb_geof(geof.unsqueeze(dim = 0))
        return geof_mlp

class SVDHead(nn.Module):
    def __init__(self, config):
        super(SVDHead, self).__init__()
        self.emb_dims = config.emb_dims
        self.reflect = nn.Parameter(torch.eye(3), requires_grad=False)
        self.reflect[2, 2] = -1
        self.alpha_factor = config.alpha_factor
        self.eps = config.eps
        self.T_net = nn.Sequential(nn.Linear(self.emb_dims, 128),
                                   nn.BatchNorm1d(128),
                                   nn.ReLU(),
                                   nn.Linear(128, 128),
                                   nn.BatchNorm1d(128),
                                   nn.ReLU(),
                                   nn.Linear(128, 128),
                                   nn.BatchNorm1d(128),
                                   nn.ReLU(),
                                   nn.Linear(128, 1),
                                   nn.ReLU())

    def forward(self, *input):
        
        src_embedding = input[0]
        tgt_embedding = input[1]
        src = input[2]
        tgt = input[3]

        batch_size = src.size(0)
        iter=input[4]
        device = src.device

        t = self.alpha_factor*torch.tensor([guess_best_alpha_torch(src_embedding[i,:], dim_num=512, transpose=True) for i in range(batch_size)], device=device)
        scores = torch.cat(
            [soft_BBS_loss_torch(src_embedding[i,:], tgt_embedding[i,:], t[i], points_dim=512, return_mat=True, transpose=True).float().unsqueeze(0)
            for i in range(batch_size)], dim=0)
        scores_norm = scores / (scores.sum(dim=2, keepdim=True)+self.eps)
        src_corr = torch.matmul(tgt, scores_norm.float().transpose(2, 1).contiguous())
        src_tgt_euc_dist = cdist_torch(src, tgt, 3)
        T = torch.clamp(self.T_net(torch.abs(src_embedding.mean(dim=2) - tgt_embedding.mean(dim=2))), 0.01, 100).view(-1,1,1)
        T = T/2**(iter-1)
        gamma = (scores * torch.exp(-src_tgt_euc_dist / T)).sum(dim=2, keepdim=True).float().transpose(2,1)
        src_weighted_mean = (src * gamma).sum(dim=2, keepdim=True) / (gamma.sum(dim=2, keepdim=True)+self.eps)
        src_centered = src - src_weighted_mean

        src_corr_weighted_mean = (src_corr * gamma).sum(dim=2, keepdim=True) / (gamma.sum(dim=2, keepdim=True) + self.eps)
        src_corr_centered = src_corr - src_corr_weighted_mean

        H = torch.matmul(src_centered * gamma, src_corr_centered.transpose(2, 1).contiguous()) + self.eps*torch.diag(torch.tensor([1,2,3], device=device)).unsqueeze(0).repeat(batch_size,1,1)

        U, S, V = [], [], []
        R = []

        for i in range(src.size(0)):
            u, s, v = torch.svd(H[i])
            r = torch.matmul(v, u.transpose(1, 0).contiguous())
            r_det = torch.det(r)
            if r_det < 0:
                u, s, v = torch.svd(H[i])
                v = torch.matmul(v, self.reflect)
                r = torch.matmul(v, u.transpose(1, 0).contiguous())
            R.append(r)

            U.append(u)
            S.append(s)
            V.append(v)

        R = torch.stack(R, dim=0)
        t = torch.matmul(-R, src_weighted_mean) + src_corr_weighted_mean
        return R, t.view(batch_size, 3), src_corr

class PoseEstimator(ME.MinkowskiNetwork):
    def __init__(self, config):
        super().__init__(D=3)

        self.config = config
        self.voxel = config.voxel_size
        self.full_matcher = FullPointMatcher(self.config)
        self.geoffeat_extraction = GeoFeatureExtraction(self.config)
        self.head = SVDHead(self.config)

    @torch.no_grad()
    def estimate_rot_trans(self, x, y, w):

        # L1 normalisation
        if self.N is not None:
           val, ind = torch.topk(w, self.N, dim=-1)
           w = torch.zeros_like(w)
           w.scatter_(-1, ind, val)
        if self.threshold is not None:
           w = w * (w > self.threshold).float()
        w = F.normalize(w, dim=-1, p=1)

        # Center point clouds
        mean_x = (w * x).sum(dim=-1, keepdim=True)
        mean_y = (w * y).sum(dim=-1, keepdim=True)
        x_centered = x - mean_x
        y_centered = y - mean_y

        # Covariance
        cov = torch.bmm(y_centered, (w * x_centered).transpose(1, 2))

        # Rotation
        U, _, V = torch.svd(cov)
        #det = torch.det(U) * torch.det(V)
        det = det_3x3(U) * det_3x3(V)
        S = torch.eye(3, device=U.device).unsqueeze(0).repeat(x.shape[0], 1, 1)
        S[:, -1, -1] = det
        R = torch.bmm(U, torch.bmm(S, V.transpose(1, 2)))

        # Translation
        T = mean_y - torch.bmm(R, mean_x)

        return R, T, w

    def forward(self, full_pcd0, full_pcd1, pcd0, pcd1, index0, index1):
        
        full_out0, full_out1 = self.full_matcher(full_pcd0, full_pcd1)
        geo_out0 = self.geoffeat_extraction(pcd0)
        geo_out1 = self.geoffeat_extraction(pcd1)

        full_out0[index0] += geo_out0.cuda()
        full_out1[index1] += geo_out1.cuda()

        rotation_ab, translation_ab, src_corr = self.head(full_out0, full_out1, full_pcd0, full_pcd1, 1)


        return full_out0, full_out1, rotation_ab, translation_ab, src_corr