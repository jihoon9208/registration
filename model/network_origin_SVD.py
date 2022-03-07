import sys
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
import teaserpp_python
import open3d as o3d
from scipy.spatial import cKDTree

from model.resunet import SparseResNetFull, SparseResNetOver
from lib.correspondence import find_correct_correspondence
from tools.pointcloud import draw_registration_result, get_matching_indices, make_open3d_point_cloud
from tools.model_util import soft_BBS_loss_torch, guess_best_alpha_torch, cdist_torch, square_dist_torch
from tools.radius import compute_graph_nn
from tools.transforms import sample_points
from tools.utils import to_tsfm
from model.common import get_norm
from model.residual_block import get_block
from model.attention import GCN, GCNOver
import tools.transform_estimation as te

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


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                           tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.generator(self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask))


class Generator(nn.Module):
    def __init__(self, emb_dims):
        super(Generator, self).__init__()
        self.nn = nn.Sequential(nn.Linear(emb_dims, emb_dims // 2),
                                nn.BatchNorm1d(emb_dims // 2),
                                nn.ReLU(),
                                nn.Linear(emb_dims // 2, emb_dims // 4),
                                nn.BatchNorm1d(emb_dims // 4),
                                nn.ReLU(),
                                nn.Linear(emb_dims // 4, emb_dims // 8),
                                nn.BatchNorm1d(emb_dims // 8),
                                nn.ReLU())
        self.proj_rot = nn.Linear(emb_dims // 8, 4)
        self.proj_trans = nn.Linear(emb_dims // 8, 3)

    def forward(self, x):
        x = self.nn(x.max(dim=1)[0])
        rotation = self.proj_rot(x)
        translation = self.proj_trans(x)
        rotation = rotation / torch.norm(rotation, p=2, dim=1, keepdim=True)
        return rotation, translation


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout=None):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)

    def forward(self, x, sublayer):
        return x + sublayer(self.norm(x))


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = None

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2).contiguous()
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.norm = nn.Sequential()  # nn.BatchNorm1d(d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = None

    def forward(self, x):
        return self.w_2(self.norm(F.relu(self.w_1(x)).transpose(2, 1).contiguous()).transpose(2, 1).contiguous())

class PointerTransform(nn.Module):
    def __init__(self, config):
        super(PointerTransform, self).__init__()
        self.emb_dims = config.emb_dims
        self.N = 1
        self.dropout = config.dropout
        self.ff_dims = config.ff_dims
        self.n_heads = config.n_heads
        c = copy.deepcopy
        attn = MultiHeadedAttention(self.n_heads, self.emb_dims)
        ff = PositionwiseFeedForward(self.emb_dims, self.ff_dims, self.dropout)
        self.model = EncoderDecoder(Encoder(EncoderLayer(self.emb_dims, c(attn), c(ff), self.dropout), self.N),
                                    Decoder(DecoderLayer(self.emb_dims, c(attn), c(attn), c(ff), self.dropout), self.N),
                                    nn.Sequential(),
                                    nn.Sequential(),
                                    nn.Sequential())

    def forward(self, src_feat, tgt_feat):
        
        src = src_feat.contiguous()
        tgt = tgt_feat.contiguous()

        tgt_embedding = self.model(src, tgt, None, None).contiguous()
        src_embedding = self.model(tgt, src, None, None).contiguous()

        return src_embedding, tgt_embedding


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

        """ gamma1 = torch.abs(gamma).sum(dim=1, keepdim=True)
        gamma_norm = gamma / (gamma1+self.eps)
        mean_src = (gamma_norm * sub_src).sum(dim=0, keepdim=True)
        mean_tgt = (gamma_norm * sub_tgt).sum(dim=0, keepdim=True)
        Sxy = torch.matmul( (sub_tgt - mean_tgt), gamma_norm * ( sub_src - mean_src).transpose(1,0))
        Sxy = Sxy.cpu().double()
        U, D, V = Sxy.svd()
        #condition = D.max(dim=1)[0] / D.min(dim=1)[0]
        S = torch.eye(3).repeat(1,1).double()
        UV_det = U.det() * V.det()
        S[2:3, 2:3] = UV_det.view(1,1)
        svT = torch.matmul( S, V.transpose(0,1) )
        R = torch.matmul( U, svT).float().to(device)
        t = mean_tgt - torch.matmul( R, mean_src ) """

        return r, t.view(-1,3), src_corr, sel0, sel1


class PoseEstimator(ME.MinkowskiNetwork):
    def __init__(self, config):
        super().__init__(D=3)

        self.config = config
        self.voxel_size = config.voxel_size
        self.full_embed = EmbeddingFeatureFull(self.config)
        self.over_embed = EmbeddingFeatureOver(self.config)
        self.embed_pointer = PointerTransform(self.config)
        self.head = SVDHead(self.config)
        
    
    def apply_transform(self, pts, trans):
        R = trans[:3, :3]
        T = trans[:3, 3]
        return pts @ R.T + T

    def find_pairs(self, F0, F1, len_batch):

        feat1tree = cKDTree(F1)
        dists, nn_inds = feat1tree.query(F0, k=32, n_jobs=-1)

        return nn_inds
    
    def evaluate_hit_ratio(self, xyz0, xyz1, T_gth, thresh=0.1):
        xyz0 = self.apply_transform(xyz0, T_gth)
        dist = np.sqrt(((xyz0 - xyz1)**2).sum(1) + 1e-6)
        return (dist < thresh).float().mean().item()

    def forward(self, full_pcd0, full_pcd1, over_pcd0, over_pcd1, over_xyz0, over_xyz1, over_index0, over_index1, inds_batch, transform, pos_pairs):
        
        xyz0 = full_pcd0.C[over_index0]
        full_out0, full_out1 = self.full_embed(full_pcd0, full_pcd1, inds_batch)
        #over_out0, over_out1 = self.over_embed(over_pcd0, over_pcd1, inds_batch)

        """ full_out0[over_index0] += over_out1
        full_out1[over_index1] += over_out0 """

        full_over_feat0 = full_out0[over_index0,:]
        full_over_feat1 = full_out1[over_index1,:]
        
        # Sample Correspondencs
        pairs, combinations = self.sample


        rotation_ab, translation_ab, src_corr, src_sel, tgt_sel = self.head(full_out0, full_out1, \
            full_pcd0.C[:,1:] * self.voxel_size, full_pcd1.C[:,1:] * self.voxel_size, transform )

        return full_out0, full_out1, rotation_ab, translation_ab, src_corr.transpose(1,0), src_sel, tgt_sel

