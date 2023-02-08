import torch

import gc
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import torch.nn as nn
import torch.nn.functional as F

import MinkowskiEngine as ME

from torch_batch_svd import svd as fast_svd

from model.resunet import ResUNetBN2C
from model.simpleunet import SimpleNet

from tools.radius import feature_matching
from tools.transform_estimation import sparse_gaussian, axis_angle_to_rotation, rotation_to_axis_angle
from lib.timer import random_quad, random_triplet
from model.feature import FCGF, FPFH ,Predator
from tools.utils import  square_distance



class PoseEstimator(ME.MinkowskiNetwork):
    def __init__(self, config):
        super().__init__(D=3)

        self.config = config
        self.voxel_size = config.voxel_size
        self.num_trial = config.num_trial
        self.r_binsize = config.r_binsize
        self.t_binsize = config.t_binsize
        self.smoothing = True
        self.kernel_size = 5

        self.feature_extraction = FCGF(self.config)
        #self.feature_extraction = FPFH(self.voxel_size)
        #self.feature_extraction = Predator(self.config)

    def sample_correspondence(self, src, tgt, src_feat, tgt_feat):

        N,_ = src.shape

        pairs_straight = feature_matching(src_feat, tgt_feat, mutual=False)
        pairs_inverse = feature_matching(tgt_feat, src_feat, mutual=False)

        pairs = torch.cat([pairs_straight, pairs_inverse.roll(1, 1)], dim=0)

        src_corr = src[pairs[:, 0]]
        tgt_corr = tgt[pairs[:, 1]]

        src_dist = ((src_corr - src_corr.roll(1, 1)).pow(2)).sum(-1).sqrt()
        tgt_dist = ((tgt_corr - tgt_corr.roll(1, 1)).pow(2)).sum(-1).sqrt()
        cross_dist = torch.abs(src_dist - tgt_dist)

        ##############################################
        # First order 

        cross_inv = cross_dist.pow(-1)
        _, index = torch.topk(cross_inv, k=int(N), dim=0, sorted=False)

        cross_inv = cross_inv[index.squeeze(dim=-1)]
        cross_inv_sum = cross_inv.sum(dtype=torch.int64)
        cross_inv_div = cross_inv / cross_inv_sum
        test = cross_inv_div.sum()

        if (cross_inv_div.all()==0):
            choice = np.random.permutation(pairs.size(0))[:int(round(N/2))]
            pairs = pairs[choice]
            masked_pairs = pairs
        else : 
            _, index = torch.topk(cross_inv_div, k=int(round(N/2)), dim=0, sorted=False)
        #masked_cross_dist = (cross_dist_inv > 0.0001).nonzero(as_tuple=False)
            masked_pairs = pairs[index.squeeze(dim=-1)]
            
        ################################
        #
        quartet = random_quad(len(masked_pairs), self.num_trial * 5)

        # check geometric constraints
        idx0 = masked_pairs[quartet, 0]
        idx1 = masked_pairs[quartet, 1]
     
        xyz0_sel = src[idx0].reshape(-1, 4, 3)
        xyz1_sel = tgt[idx1].reshape(-1, 4, 3)

        dist = square_distance(xyz0_sel, xyz1_sel)

        index = torch.arange(4).expand(dist.shape[0],4).reshape(dist.shape[0],1,4)

        dist_diagonal = torch.gather(dist, 1, index.to(dist.device) )
        dist_diagonal = dist_diagonal.type(torch.DoubleTensor).pow(-1)
        _, index = torch.topk(dist_diagonal, k = 3, dim = -1, sorted=False)
        index = index.squeeze(dim=1)

        triplets = torch.gather(quartet, 1 ,index)

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

        if triplets.shape[0] > (self.num_trial / 10):
            idx = np.round(
                np.linspace(0, triplets.shape[0] - 1, int(self.num_trial / 10))
            ).astype(int)
            triplets = triplets[idx]

        return masked_pairs, triplets, cross_inv_div


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
        translate = xyz1_mean.squeeze(1) - xyz0_rotated

        return angles, translate

    def feature_measure(self, feat0, feat1, pairs, triplets):

        feat0_sel = feat0[pairs[triplets, 0]]
        feat1_sel = feat1[pairs[triplets, 1]]

        feat0_mean = feat0_sel.mean(1, keepdim=True).squeeze(dim=1)
        feat1_mean = feat1_sel.mean(1, keepdim=True).squeeze(dim=1)

        N, _ = feat0_mean.shape
        M, _ = feat1_mean.shape

        dist = -2 * torch.matmul(feat0_mean, feat1_mean.T)     
        dist += torch.sum(feat0_mean ** 2, -1).view(N, 1)
        dist += torch.sum(feat1_mean ** 2, -1).view(1, M)

        diag_ind = range(feat1_mean.shape[0])
        dist[diag_ind,diag_ind] = np.inf
        final_dist = dist.min(dim=1).values

        return final_dist

    def vote(self, Rs, ts, dist):
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
        #feat = torch.ones(coord.shape[0]).unsqueeze(1).to(r_coord.device)
        feat = dist.unsqueeze(1)
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

    def forward(self, full_pcd0, full_pcd1, over_xyz0, over_xyz1, over_index0, over_index1 , refine_model):
        
        global rotation
        global translate

        gc.collect()

        # FCGF
        full_out0 = self.feature_extraction.extract_feature(full_pcd0)
        full_out1 = self.feature_extraction.extract_feature(full_pcd1)

        # FHFP
        #full_out0, xyz0 = self.feature_extraction.extract_feature((full_pcd0.C[:,1:] * self.voxel_size).detach().cpu().numpy())
        #full_out1, xyz1 = self.feature_extraction.extract_feature((full_pcd1.C[:,1:] * self.voxel_size).detach().cpu().numpy())

        # Predator
        #full_out0, full_out1 = self.feature_extraction.extract_feature(full_pcd0, full_pcd1)
        
        full_over_feat0 = full_out0[over_index0,:].to(full_pcd0.device)
        full_over_feat1 = full_out1[over_index1,:].to(full_pcd0.device)
        
        pairs, combinations, confidence = self.sample_correspondence(over_xyz0, over_xyz1, full_over_feat0, full_over_feat1)
        angles, ts = self.solve(over_xyz0, over_xyz1, pairs, combinations)

        dist = self.feature_measure(full_over_feat0, full_over_feat1, pairs, combinations)

        # rotation & translation voting
        votes = self.vote(angles, ts, dist)

        # gaussian smoothing
        if self.smoothing:
            votes = sparse_gaussian(
                votes, kernel_size=self.kernel_size, dimension=6
            )

        # post processing
        #if self.refine_model is not None:
        #    votes = self.refine_model(votes)

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
        
        return Transform, votes, confidence

