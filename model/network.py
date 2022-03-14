import sys
import logging
from turtle import forward
from matplotlib.transforms import Transform
import torch
import math
import numpy as np
from typing import Any
import torch.nn as nn
import torch.nn.functional as F
import copy
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MEF
from torch_batch_svd import svd as fast_svd

from model.resunet import SparseResNetFull, SparseResNetOver
from model.simpleunet import SimpleNet

from tools.pointcloud import draw_registration_result
from tools.model_util import soft_BBS_loss_torch, guess_best_alpha_torch, cdist_torch
from tools.radius import compute_graph_nn, feature_matching
from tools.transform_estimation import sparse_gaussian, axis_angle_to_rotation, rotation_to_axis_angle
from lib.timer import random_triplet

from model.attention import GCN, GCNOver

sys.path.append("./lib/model_exc")

import libply_c

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1).contiguous()) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    return torch.matmul(p_attn, value), p_attn

def point_permute(p):
    return p.permute(1,0)
  
def xyz_permute(xyz):
    return xyz.permute(1,0)


class Aggregation(ME.MinkowskiNetwork):

    CHANNELS = [None, 32, 64, 128, 256, 512]
    def __init__(self, config, D=3):

        ME.MinkowskiNetwork.__init__(self, D)
        channels = self.CHANNELS
        self.layer = torch.nn.Sequential(torch.nn.Linear(channels[1] * 2, channels[1] * 2), 
                    torch.nn.LayerNorm(channels[1] * 2),
                    torch.nn.ReLU(), 
                    torch.nn.Linear(channels[1] * 2, channels[1]))

    def forward(self, sparse, geof):
        
        d_concate = torch.cat([sparse, geof], dim=-1).cuda()

        feat_concat = self.layer(d_concate)
        
        return feat_concat 

class GeoFeatureExtraction(ME.MinkowskiNetwork):
    def __init__(self, config):
        
        super().__init__(D=3)
        self.config = config
        self.k_nn_geof = config.k_nn_geof
        self.voxel_size = config.voxel_size
        self.batch_size = config.batch_size
        self.emb_dims = 32
        self.emb_geof = DGCNN(emb_dims=self.emb_dims)
      

    def forward(self, pcd0, pcd1):
        #
        # PCD N * 3
        # geof N * D
        graph_nn0 = compute_graph_nn(pcd0, self.k_nn_geof)
        geof0 = libply_c.compute_norm(pcd0, graph_nn0["target"], self.k_nn_geof).astype('float32')
        geof0 = torch.from_numpy(geof0).detach() 
        geof_mlp0 = self.emb_geof(geof0.unsqueeze(dim = 0)).squeeze(dim=0)

        graph_nn1 = compute_graph_nn(pcd1, self.k_nn_geof)
        geof1 = libply_c.compute_norm(pcd1, graph_nn1["target"], self.k_nn_geof).astype('float32')
        geof1 = torch.from_numpy(geof1).detach() 
        geof_mlp1 = self.emb_geof(geof1.unsqueeze(dim = 0)).squeeze(dim=0)

        return geof_mlp0.transpose(1,0) ,geof_mlp1.transpose(1,0)

class EmbeddingFeatureFull(ME.MinkowskiNetwork):
    def __init__(self, config):
        self.config = config
        super().__init__(D=3)

        self.voxel_size = config.voxel_size
        self.aggregation = Aggregation(config)
        num_feats = 1

        self.SparseResNet = SparseResNetFull(
            self.config, 
            num_feats,
            config.sparse_dims,
            bn_momentum=config.bn_momentum,
            conv1_kernel_size=config.conv1_kernel_size,
            D=3).cuda()

        self.GCN = GCN(self.config)

        self.geoffeat_extraction = GeoFeatureExtraction(self.config)

    def forward(self, full_pcd0, full_pcd1, inds_batch):
        
        ###############################
        # FCGF
        sa_feat0 = self.SparseResNet(full_pcd0)
        sa_feat1 = self.SparseResNet(full_pcd1)

        ###############################
        # Attention

        xyz_batch0 = full_pcd0.C[:,1:] * self.voxel_size
        xyz_batch1 = full_pcd1.C[:,1:] * self.voxel_size

        attention0 = self.GCN(xyz_batch0, sa_feat0)
        attention1 = self.GCN(xyz_batch1, sa_feat1)

        geo_out0, geo_out1 = self.geoffeat_extraction(xyz_batch0.detach().cpu().numpy(), xyz_batch1.detach().cpu().numpy())        

        total_feat0 = torch.cat([sa_feat0, attention0, geo_out0], dim=-1)
        total_feat1 = torch.cat([sa_feat1, attention1, geo_out1], dim=-1)

        return total_feat0, total_feat1

class EmbeddingFeatureOver(ME.MinkowskiNetwork):
    def __init__(self, config):
        self.config = config
        super().__init__(D=3)

        self.voxel_size = config.voxel_size
        self.aggregation = Aggregation(config)
        num_feats = 1

        self.SparseResNet = SparseResNetOver(
            self.config, 
            num_feats,
            config.sparse_dims,
            bn_momentum=config.bn_momentum,
            conv1_kernel_size=config.conv1_kernel_size,
            D=3).cuda()

        self.GCN = GCNOver(self.config)

        self.geoffeat_extraction = GeoFeatureExtraction(self.config)

    def forward(self, over_pcd0, over_pcd1, inds_batch):
        
        ###############################
        # FCGF
        sa_feat0 = self.SparseResNet(over_pcd0)
        sa_feat1 = self.SparseResNet(over_pcd1)

        ###############################
        # Attention
        xyz_batch0 = over_pcd0.C[:,1:] * self.voxel_size
        xyz_batch1 = over_pcd1.C[:,1:] * self.voxel_size

        attention0 = self.GCN(xyz_batch0, sa_feat0)
        attention1 = self.GCN(xyz_batch1, sa_feat1)

        geo_out0, geo_out1 = self.geoffeat_extraction(xyz_batch0.detach().cpu().numpy(), xyz_batch1.detach().cpu().numpy())        
        
        total_feat0 = torch.cat([sa_feat0, attention0, geo_out0], dim=-1)
        total_feat1 = torch.cat([sa_feat1, attention1, geo_out1], dim=-1)

        return total_feat0, total_feat1

class DGCNN(nn.Module):
    def __init__(self, emb_dims=512):
        super(DGCNN, self).__init__()
        self.conv1 = nn.Conv1d(4, 32, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(128, 64, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(288, emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(64)
        self.bn5 = nn.BatchNorm1d(emb_dims)

    def forward(self, x):
        x = x.transpose(2,1).cuda()
        batch_size, num_dims, num_points = x.size()

        x = F.relu(self.bn1(self.conv1(x)))
        x1 = x.max(dim=0, keepdim=True)[0]

        x = F.relu(self.bn2(self.conv2(x)))
        x2 = x.max(dim=0, keepdim=True)[0]

        x = F.relu(self.bn3(self.conv3(x)))
        x3 = x.max(dim=0, keepdim=True)[0]

        x = F.relu(self.bn4(self.conv4(x)))
        x4 = x.max(dim=0, keepdim=True)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = F.relu(self.bn5(self.conv5(x))).view(batch_size, -1, num_points)
        return x

class DGCNN_over(nn.Module):
    def __init__(self, config):
        super(DGCNN_over, self).__init__()

        self.emb_dims = config.emb_dims * 3
        self.conv1 = nn.Conv1d(3, 32, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(128, 64, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(288, self.emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(64)
        self.bn5 = nn.BatchNorm1d(self.emb_dims)

    def forward(self, x, y):
        pcd0 = x.unsqueeze(dim=0).transpose(2,1).cuda()
        pcd1 = y.unsqueeze(dim=0).transpose(2,1).cuda()
        batch_size, num_dims, num_points0 = pcd0.size()
        batch_size, num_dims, num_points1 = pcd1.size()

        #########################################

        pcd_out0 = F.relu(self.bn1(self.conv1(pcd0)))
        pcd_max01 = pcd_out0.max(dim=0, keepdim=True)[0]

        pcd_out0 = F.relu(self.bn2(self.conv2(pcd_out0)))
        pcd_max02 = pcd_out0.max(dim=0, keepdim=True)[0]

        pcd_out0 = F.relu(self.bn3(self.conv3(pcd_out0)))
        pcd_max03 = pcd_out0.max(dim=0, keepdim=True)[0]

        pcd_out0 = F.relu(self.bn4(self.conv4(pcd_out0)))
        pcd_max04 = pcd_out0.max(dim=0, keepdim=True)[0]

        pcd_out0 = torch.cat((pcd_max01, pcd_max02, pcd_max03, pcd_max04), dim=1)

        pcd_out0 = F.relu(self.bn5(self.conv5(pcd_out0))).view(batch_size, -1, num_points0)

        ######################################
        # 
        pcd_out1= F.relu(self.bn1(self.conv1(pcd1)))
        pcd_max11 = pcd_out1.max(dim=0, keepdim=True)[0]

        pcd_out1 = F.relu(self.bn2(self.conv2(pcd_out1)))
        pcd_max12 = pcd_out1.max(dim=0, keepdim=True)[0]

        pcd_out1 = F.relu(self.bn3(self.conv3(pcd_out1)))
        pcd_max13 = pcd_out1.max(dim=0, keepdim=True)[0]

        pcd_out1 = F.relu(self.bn4(self.conv4(pcd_out1)))
        pcd_max14 = pcd_out1.max(dim=0, keepdim=True)[0]

        pcd_out1 = torch.cat((pcd_max11, pcd_max12, pcd_max13, pcd_max14), dim=1)

        pcd_out1 = F.relu(self.bn5(self.conv5(pcd_out1))).view(batch_size, -1, num_points1)

        return pcd_out0.squeeze(dim=0).transpose(1,0), pcd_out1.squeeze(dim=0).transpose(1,0)

class TNet(nn.Module):
    def __init__(self, config): 
        super(TNet, self).__init__()
        self.emb_dims = config.emb_dims * 3

        self.layer1 = nn.Sequential(nn.Linear(self.emb_dims, 16),
                                   nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(16, 16),
                                 
                                   nn.ReLU())
        self.layer3 = nn.Sequential(nn.Linear(16, 32),
                                  
                                   nn.ReLU())
        self.layer4 = nn.Sequential(nn.Linear(32, 1),
                                   nn.ReLU())

    def forward(self, input):
        out = self.layer1(input)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        return out

class SVDHead(nn.Module):
    def __init__(self, config):
        super(SVDHead, self).__init__()
        
        self.reflect = nn.Parameter(torch.eye(3), requires_grad=False)
        self.reflect[2, 2] = -1
        self.alpha_factor = config.alpha_factor
        self.num_hn_samples = config.sample_num_points
        self.epsilon = torch.nn.Parameter(torch.tensor(-5.0))
        self.eps = config.eps
        self.T_net = TNet(config)
        """ self.T_net = nn.Sequential(nn.Linear(self.emb_dims, 16),
                                   nn.BatchNorm1d(16),
                                   nn.ReLU(),
                                   nn.Linear(16, 16),
                                   nn.BatchNorm1d(16),
                                   nn.ReLU(),
                                   nn.Linear(16, 32),
                                   nn.BatchNorm1d(32),
                                   nn.ReLU(),
                                   nn.Linear(32, 1),
                                   nn.ReLU()) """
    def apply_transform(self, pts, trans):
        R = trans[:3, :3]
        T = trans[:3, 3]
        return pts @ R.T + T
        #return pts @ rot.T + trans.T

    def forward(self, src_feat, tgt_feat, src, tgt, trans):

        #####################################
        # src_embedding N x D
        # tgt_embedding N x D
        N0, N1 = len(src_feat), len(tgt_feat)

        src_embedding = src_feat.transpose(1,0).cuda()
        tgt_embedding = tgt_feat.transpose(1,0).cuda()

        src = src.transpose(1,0).cuda()
        tgt = tgt.transpose(1,0).cuda()

        sel0 = np.random.choice(N0, min(N0, self.num_hn_samples), replace=False)
        sel1 = np.random.choice(N1, min(N1, self.num_hn_samples), replace=False)

        # Find negatives for all F1[positive_pairs[:, 1]]
        sub_feat0, sub_feat1 = src_embedding[:, sel0], tgt_embedding[:, sel1]
        sub_src, sub_tgt = src[:, sel0], tgt[:, sel1]

        iter=1
        device = src.device

        t = self.alpha_factor * torch.tensor([guess_best_alpha_torch(sub_feat0, dim_num=96, transpose=True)], device=device)
        scores = soft_BBS_loss_torch(sub_feat0, sub_feat1, t, points_dim=96, return_mat=True, transpose=True).float()
        scores_norm = scores / (scores.sum(dim=1, keepdim=True) + self.eps)
        src_corr = torch.matmul(sub_tgt, scores_norm.float().transpose(1,0).contiguous())

        src_tgt_euc_dist = cdist_torch(sub_src, sub_tgt, 3)
        
        T = self.T_net(torch.abs((sub_feat0.mean(dim=1).unsqueeze(dim=0) - sub_feat1.mean(dim=1).unsqueeze(dim=0))))
        T = torch.clamp(self.T_net(torch.abs((sub_feat0.mean(dim=1).unsqueeze(dim=0) - sub_feat1.mean(dim=1).unsqueeze(dim=0)))), 0.01, 100).view(-1,1)
        T = T/2**(iter-1)
        gamma = (scores * torch.exp(-src_tgt_euc_dist / T)).sum(dim=1, keepdim=True).float().transpose(1,0)
        #gamma = (scores * torch.exp(-src_tgt_euc_dist)).sum(dim=1, keepdim=True).float().transpose(1,0)

        src_weighted_mean = (sub_src * gamma).sum(dim=1, keepdim=True) / (gamma.sum(dim=1, keepdim=True) + self.eps)
        src_centered = sub_src - src_weighted_mean

        src_corr_weighted_mean = (src_corr * gamma).sum(dim=1, keepdim=True) / (
                    gamma.sum(dim=1, keepdim=True) + self.eps)
        src_corr_centered = src_corr - src_corr_weighted_mean

        H = torch.matmul( src_centered * gamma, src_corr_centered.transpose(1, 0).contiguous()) 
        H0 = self.eps * torch.diag(torch.tensor([1, 2, 3], device=device)).repeat( 1, 1)
        H += H0
        
        u, s, v = torch.svd(H)
        r = torch.matmul(v, u.transpose(1, 0).contiguous())
        r_det = torch.det(r)
        if r_det < 0:
            u, s, v = torch.svd(H)
            v = torch.matmul(v, self.reflect)
            r = torch.matmul(v, u.transpose(1, 0).contiguous())

        t = torch.matmul(-r, src_weighted_mean) + src_corr_weighted_mean

        return r, t.view(-1,3), src_corr, sel0, sel1


class PoseEstimator(ME.MinkowskiNetwork):
    def __init__(self, config):
        super().__init__(D=3)

        self.config = config
        self.voxel_size = config.voxel_size
        self.num_trial = config.num_trial
        self.r_binsize = config.r_binsize
        self.t_binsize = config.t_binsize
        self.smoothing = True
        self.kernel_size = 3
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.full_embed = EmbeddingFeatureFull(self.config)
        self.refine_model = SimpleNet (
            conv1_kernel_size=config.conv1_kernel_size,
            D=6).cuda()

        self.head = SVDHead(self.config)
        
    def sample_correspondence(self, src, tgt, src_feat, tgt_feat):

        pairs = feature_matching(src_feat, tgt_feat, mutual=False)
        pairs_inv = feature_matching(tgt_feat, src_feat, mutual=False)
        pairs = torch.cat([pairs, pairs_inv.roll(1, 1)], dim=0)

        # sample random triplets
        triplets = random_triplet(len(pairs), self.num_trial * 5)

        # check geometric constraints
        idx0 = pairs[triplets, 0]
        idx1 = pairs[triplets, 1]
        xyz0_sel = src[idx0].reshape(-1, 3, 3)
        xyz1_sel = tgt[idx1].reshape(-1, 3, 3)
        li = torch.norm(xyz0_sel - xyz0_sel.roll(1, 1), p=2, dim=2)
        lj = torch.norm(xyz1_sel - xyz1_sel.roll(1, 1), p=2, dim=2)

        triangle_check = torch.all(
            torch.abs(li - lj) < 3 * self.voxel_size, dim=1
        ).cpu()
        dup_check = torch.logical_and(
            torch.all(li > self.voxel_size * 1.5, dim=1),
            torch.all(lj > self.voxel_size * 1.5, dim=1),
        ).cpu()

        triplets = triplets[torch.logical_and(triangle_check, dup_check)]

        if triplets.shape[0] > self.num_trial:
            idx = np.round(
                np.linspace(0, triplets.shape[0] - 1, self.num_trial)
            ).astype(int)
            triplets = triplets[idx]

        return pairs, triplets

    def apply_transform(self, pts, trans):
        R = trans[:3, :3]
        T = trans[:3, 3]
        return pts @ R.T + T

    def solve(self, xyz0, xyz1, pairs, triplets):
        xyz0_sel = xyz0[pairs[triplets, 0]]
        xyz1_sel = xyz1[pairs[triplets, 1]]

        # zero mean shift
        xyz0_mean = xyz0_sel.mean(1, keepdim=True)
        xyz1_mean = xyz1_sel.mean(1, keepdim=True)
        xyz0_centered = xyz0_sel - xyz0_mean
        xyz1_centered = xyz1_sel - xyz1_mean

        # solve rotation
        H = xyz1_centered.transpose(1, 2) @ xyz0_centered
        U, D, V = fast_svd(H)
        S = torch.eye(3).repeat(U.shape[0], 1, 1).to(U.device)
        det = U.det() * V.det()
        S[det < 0, -1, -1] = -1
        Rs = U @ S @ V.transpose(1, 2)
        angles = rotation_to_axis_angle(Rs)

        # solve translation using centroid
        xyz0_rotated = torch.bmm(Rs, xyz0_mean.permute(0, 2, 1)).squeeze(2)
        t = xyz1_mean.squeeze(1) - xyz0_rotated

        return angles, t

    def vote(self, Rs, ts):
        r_coord = torch.floor(Rs / self.r_binsize)
        t_coord = torch.floor(ts / self.t_binsize)
        coord = torch.cat(
            [
                torch.zeros(r_coord.shape[0]).unsqueeze(1).to(self.device),
                r_coord,
                t_coord,
            ],
            dim=1,
        )
        feat = torch.ones(coord.shape[0]).unsqueeze(1).to(self.device)
        vote = ME.SparseTensor(
            feat.float(),
            coordinates=coord.int(),
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_SUM,
        )
        return vote
    
    def evaluate(self, vote):
        max_index = vote.F.squeeze(1).argmax()
        max_value = vote.C[max_index, 1:]
        angle = (max_value[:3] + 0.5) * self.r_binsize
        t = (max_value[3:] + 0.5) * self.t_binsize
        R = axis_angle_to_rotation(angle)
        return R, t

    #def forward(self, full_pcd0, full_pcd1, over_pcd0, over_pcd1, over_xyz0, over_xyz1, over_index0, over_index1, inds_batch, transform, pos_pairs):
    def forward(self, full_pcd0, full_pcd1, over_xyz0, over_xyz1, over_index0, over_index1, inds_batch, transform, pos_pairs):
        
        global Rotation
        global translate
        
        full_out0, full_out1 = self.full_embed(full_pcd0, full_pcd1, inds_batch)
        #over_out0, over_out1 = self.over_embed(over_pcd0, over_pcd1, inds_batch)

        full_over_feat0 = full_out0[over_index0,:]
        full_over_feat1 = full_out1[over_index1,:]
        
        # Sample Correspondencs
        pairs, combinations = self.sample_correspondence(over_xyz0, over_xyz1, full_over_feat0, full_over_feat1)

      
        angles, ts = self.solve(over_xyz0, over_xyz1, pairs, combinations)

        # rotation & translation voting
        votes = self.vote(angles, ts)

        # gaussian smoothing
        if self.smoothing:
            votes = sparse_gaussian(
                votes, kernel_size=self.kernel_size, dimension=6
            )

        # post processing
        if self.refine_model is not None:
            votes = self.refine_model(votes)

        Rotation, translate = self.evaluate(votes)
        self.hspace = votes
        Transform = torch.eye(4)
        Transform[:3, :3] = Rotation
        Transform[:3, 3] = translate
        # empty cache
        torch.cuda.empty_cache()

   
        """ import pdb
        pdb.set_trace()
        return torch.eye(4) """

        """ rotation_ab, translation_ab, src_corr, src_sel, tgt_sel = self.head(full_out0, full_out1, \
            full_pcd0.C[:,1:] * self.voxel_size, full_pcd1.C[:,1:] * self.voxel_size, transform ) """

        return full_over_feat0, full_over_feat1, Rotation, translate

