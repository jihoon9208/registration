import sys
import logging
from turtle import clear, forward
from matplotlib.transforms import Transform
import torch
import math
import gc
import numpy as np
from typing import Any
import torch.nn as nn
import torch.nn.functional as F
import copy
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MEF
from torch_batch_svd import svd as fast_svd
from lib.sparse import corr_and_add

from model.resunet import SparseResNet
from model.simpleunet import SimpleNet
from tools.pointcloud import draw_registration_result
from tools.radius import compute_graph_nn, feature_matching, feat_match
from tools.transform_estimation import sparse_gaussian, axis_angle_to_rotation, rotation_to_axis_angle
from lib.timer import random_triplet

from model.attention import GCN
from tools.transforms import apply_transform
from tools.utils import to_o3d_pcd

sys.path.append("./lib/model_exc")

#import libply_c

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

class EmbeddingFeatureFull(ME.MinkowskiNetwork):
    def __init__(self, config):
        self.config = config
        super().__init__(D=3)

        self.voxel_size = config.voxel_size
    
        num_feats = 1

        self.SparseResNet = SparseResNet(
            self.config, 
            num_feats,
            config.sparse_dims,
            bn_momentum=config.bn_momentum,
            conv1_kernel_size=config.conv1_kernel_size,
            D=3).cuda()

        #self.GCN = GCN(self.config)
        self.emb_dims = 32


        self.layer1 = nn.Sequential(
            nn.Linear(1088, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(),
        )
        self.layer3 = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
        )
        self.layer4 = nn.Sequential(
            nn.Linear(128, 32)  
        )

    def make_mask (self, feat_out, x):
        _, feat_dim = feat_out.size()
        x = x.repeat(1, feat_dim)
        total_feat = torch.cat([feat_out, x ], dim =-1)
        mask = self.layer2(total_feat)

        return mask

    def index_points(self, points, idx):
        """
		Input:
			points: input points data, [B, N, C]
			idx: sample index data, [B, S]
		Return:
			new_points:, indexed points data, [B, S, C]
		"""
        device = points.device
        N, C = points.shape
        masked_points = torch.zeros(N, C, device=device)
        index0 = idx[:,0]
        index1 = idx[:,1]
        masked_points[index0, index1] = points[index0, index1]

        return masked_points

    def find_index(self, mask_val):

        mask_idx = torch.nonzero((mask_val>0.5)*1.0)
        return mask_idx

    def forward(self, full_pcd):
        
        pcd = full_pcd.C[:,1:] * self.voxel_size
        N, C = pcd.shape
        """ sparse_feat, sparse_att_feat = self.SparseResNet(full_pcd)
        sparse_feat = self.max(sparse_feat.F)
        
        mask = self.make_mask(sparse_feat, sparse_att_feat.F)

        mask_dix = self.find_index(mask)

        masked_feat = self.index_points(sparse_feat, mask_dix)
        final_feat = torch.cat([masked_feat, sparse_att_feat.T], dim=-1) """

        sparse_feat , sparse_global_feat = self.SparseResNet(full_pcd)
        #cluster_feat = self.GCN(pcd, sinput.F)
        """ reshpae_sparse_feat = sparse_global_feat.F.repeat(N,1)
        
        #dgcnn_feat , dgcnn_att_feat = self.emb_geof(pcd)
        total_feat = torch.cat([sparse_feat.F, reshpae_sparse_feat], dim=-1)

        x = self.layer1(total_feat)
        x = self.layer2(x)
        x = self.layer3(x)
        final_feat = self.layer4(x)

        final_feat = F.log_softmax(final_feat, dim=1) """
        
        
        return sparse_feat

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
        
        self.full_embed = EmbeddingFeatureFull(self.config)
        """ self.refine_model = SimpleNet (
            conv1_kernel_size=config.conv1_kernel_size,
            D=6).cuda() """
        
    def sample_correspondence(self, src, tgt, src_feat, tgt_feat):

        N,_ = src.shape

        pairs = feature_matching(src_feat, tgt_feat, mutual=False)
        pairs_inv = feature_matching(tgt_feat, src_feat, mutual=False)

        pairs = torch.cat([pairs, pairs_inv.roll(1, 1)], dim=0)

        src_corr = src[pairs[:, 0]]
        tgt_corr = tgt[pairs[:, 1]]

        src_dist = ((src_corr - src_corr.roll(1, 1)) ** 2).sum(-1) ** 0.5
        tgt_dist = ((tgt_corr - tgt_corr.roll(1, 1)) ** 2).sum(-1) ** 0.5
        cross_dist = torch.abs(src_dist - tgt_dist)

        #local_measure = (cross_dist < 4 * self.voxel_size).float()

        cross_dist_inv = cross_dist.pow(-1)
        sum_cross_dist_inv = cross_dist_inv.sum()
        cross_dist_inv = cross_dist_inv.div(sum_cross_dist_inv)

        if (cross_dist_inv.all()==0):
            masked_pairs = pairs
        else : 
            _, index = torch.topk(cross_dist_inv, k=int(round(N/2)), dim=0)
            #masked_cross_dist = (cross_dist_inv > 0.0001).nonzero(as_tuple=False)
            masked_pairs = pairs[index.squeeze(dim=-1)]
            
        
        # sample random triplets
        triplets = random_triplet(len(masked_pairs), self.num_trial * 5)

        # check geometric constraints
        idx0 = masked_pairs[triplets, 0]
        idx1 = masked_pairs[triplets, 1]
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

        return masked_pairs, triplets


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
                torch.zeros(r_coord.shape[0]).unsqueeze(1).to(r_coord.device),
                r_coord,
                t_coord,
            ],
            dim=1,
        )
        feat = torch.ones(coord.shape[0]).unsqueeze(1).to(r_coord.device)
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

    def forward(self, full_pcd0, full_pcd1, over_xyz0, over_xyz1, over_index0, over_index1, refine_model):
        
        global rotation
        global translate
        gc.collect()
        
        full_out0 = self.full_embed(full_pcd0)
        full_out1 = self.full_embed(full_pcd1)
         
        full_over_feat0 = full_out0.F[over_index0,:]
        full_over_feat1 = full_out1.F[over_index1,:]
        
        del full_out0, full_out1

        """ xyz0 = full_pcd0.C[:,1:] * self.voxel_size
        xyz1 = full_pcd1.C[:,1:] * self.voxel_size """
        
        # Sample Correspondencs

        pairs, combinations = self.sample_correspondence(over_xyz0, over_xyz1, full_over_feat0, full_over_feat1)

        del full_over_feat0, full_over_feat1

        angles, ts = self.solve(over_xyz0, over_xyz1, pairs, combinations)

        # rotation & translation voting
        votes = self.vote(angles, ts)
        del angles, ts

        # gaussian smoothing
        if self.smoothing:
            votes = sparse_gaussian(
                votes, kernel_size=self.kernel_size, dimension=6
            )

        # post processing
        if refine_model is not None:
            votes = refine_model(votes)
        
        rotation, translate = self.evaluate(votes)
        self.hspace = votes
        Transform = torch.eye(4)
        Transform[:3, :3] = rotation
        Transform[:3, 3] = translate
        # empty cache
        gc.collect()
        torch.cuda.empty_cache()
        
        #draw_registration_result(xyz0.detach().cpu().numpy(), xyz1.detach().cpu().numpy(), Transform.detach().cpu().numpy())

        return Transform, votes

