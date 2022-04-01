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

from model.resunet import SparseResNetFull
from model.simpleunet import SimpleNet
from tools.pointcloud import draw_registration_result
from tools.radius import compute_graph_nn, feature_matching
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

class GeoFeatureExtraction(ME.MinkowskiNetwork):
    def __init__(self, config):
        
        super().__init__(D=3)
        self.config = config
        self.k_nn_geof = config.k_nn_geof
        self.voxel_size = config.voxel_size
        self.batch_size = config.batch_size
        self.emb_dims = 32
        self.emb_geof = DGCNN_geo(emb_dims=self.emb_dims)
      

    def forward(self, pcd):
        #
        # PCD N * 3
        # geof N * D
        graph_nn0 = compute_graph_nn(pcd, self.k_nn_geof)
        geof = libply_c.compute_norm(pcd, graph_nn0["target"], self.k_nn_geof).astype('float32')
        geof = torch.from_numpy(geof).detach() 
        geof_mlp = self.emb_geof(geof.unsqueeze(dim = 0)).squeeze(dim=0)

        return geof_mlp.transpose(1,0)

class EmbeddingFeatureFull(ME.MinkowskiNetwork):
    def __init__(self, config):
        self.config = config
        super().__init__(D=3)

        self.voxel_size = config.voxel_size
    
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
        self.emb_dims = 32
        self.DGCNN = DGCNN(emb_dims=self.emb_dims)

        self.layer1 = nn.Sequential(
            nn.Linear(32, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.Sigmoid()
        )
        self.max =  nn.Softmax(dim=1)
   

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
        sparse_feat, sparse_att_feat = self.SparseResNet(full_pcd)
        sparse_feat = self.max(sparse_feat.F)
        
        mask = self.make_mask(sparse_feat, sparse_att_feat.F)

        mask_dix = self.find_index(mask)
        final_feat = self.index_points(sparse_feat, mask_dix)

        return final_feat

class DGCNN(nn.Module):
    def __init__(self, emb_dims=512):
        super(DGCNN, self).__init__()
        self.conv1 = nn.Conv1d(3, 32, kernel_size=1, bias=False)
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

class DGCNN_geo(nn.Module):
    def __init__(self, emb_dims=512):
        super(DGCNN_geo, self).__init__()
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
        self.refine_model = SimpleNet (
            conv1_kernel_size=config.conv1_kernel_size,
            D=6).cuda()
        
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

    def forward(self, full_pcd0, full_pcd1, over_xyz0, over_xyz1, over_index0, over_index1, inds_batch):
        
        global rotation
        global translate
        
        full_out0 = self.full_embed(full_pcd0)
        full_out1 = self.full_embed(full_pcd1)
         
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
        
        rotation, translate = self.evaluate(votes)
        self.hspace = votes
        Transform = torch.eye(4)
        Transform[:3, :3] = rotation
        Transform[:3, 3] = translate 
        # empty cache
        torch.cuda.empty_cache()
        
        #draw_registration_result(xyz0.detach().cpu().numpy(), xyz1.detach().cpu().numpy(), Transform.detach().cpu().numpy())

        """ import pdb
        pdb.set_trace()
        return torch.eye(4) """

        """ rotation_ab, translation_ab, src_corr, src_sel, tgt_sel = self.head(full_out0, full_out1, \
            full_pcd0.C[:,1:] * self.voxel_size, full_pcd1.C[:,1:] * self.voxel_size, transform ) """

        return full_over_feat0, full_over_feat1, Transform, votes

