# -*- coding: future_fstrings -*-
import torch
import sys
import MinkowskiEngine.MinkowskiOps as MEO
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MEF

from model.common import get_norm

from model.residual_block import get_block

from typing import Any
import torch.nn as nn
import torch.nn.functional as F

from model.attention import GCN

import gc

sys.path.append("./lib/partition/ply_c")



class SparseResNet(ME.MinkowskiNetwork):
  
  NORM_TYPE = 'BN'
  BLOCK_NORM_TYPE = 'BN'
  CHANNELS = [None, 32, 64, 128, 256, 512]
  TR_CHANNELS = [None, 32, 64, 64, 128]

  # To use the model, must call initialize_coords before forward pass.
  # Once data is processed, call clear to reset the model before calling initialize_coords
  def __init__(self,config,
               in_channels=3,
               out_channels=32,
               bn_momentum=0.1,
               normalize_feature=None,
               conv1_kernel_size=None,
               D=3):
    ME.MinkowskiNetwork.__init__(self, D)

    self.voxel_size = config.voxel_size
    NORM_TYPE = self.NORM_TYPE
    BLOCK_NORM_TYPE = self.BLOCK_NORM_TYPE
    CHANNELS = self.CHANNELS
    TR_CHANNELS = self.TR_CHANNELS

    self.device = torch.device('cuda')

    self.k_nn_geof = config.k_nn_geof

    self.alpha_factor = 4
    self.normalize_feature = normalize_feature

    self.maxpooling = ME.MinkowskiMaxPooling(kernel_size=2, stride=1, dimension=D)

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
        in_channels=CHANNELS[4], # ,
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

    self.conv1_tr_att = ME.MinkowskiConvolution(
        in_channels=CHANNELS[4],
        out_channels=CHANNELS[5],
        kernel_size=3,
        stride=2,
        dilation=1,
        bias=False,
        dimension=D)

    self.conv2_tr_att = ME.MinkowskiConvolution(
        in_channels=CHANNELS[5],
        out_channels=CHANNELS[5] * 2,
        kernel_size=3,
        stride=2,
        dilation=1,
        bias=False,
        dimension=D)
    
    self.final = ME.MinkowskiConvolution(
        in_channels=TR_CHANNELS[1],
        out_channels=out_channels * 2,
        kernel_size=1,
        stride=1,
        dilation=1,
        bias=True,
        dimension=D)

    self.final_att = ME.MinkowskiConvolution(
        in_channels=TR_CHANNELS[1],
        out_channels=1,
        kernel_size=1,
        stride=1,
        dilation=1,
        bias=True,
        dimension=D)
    self.glob_avg = ME.MinkowskiGlobalMaxPooling()
    self.final_att = ME.MinkowskiLinear(CHANNELS[5] * 2, CHANNELS[5] * 2, bias=True)

    self.final_att_act = torch.nn.Softplus(beta=1, threshold=20)

    
  def forward(self, x):
    
    # source feature 
    out_s1 = self.conv1(x)
    out_s1 = self.norm1(out_s1)
    out_s1 = self.block1(out_s1)
    out = MEF.relu(out_s1)

    out_s2 = self.conv2(out)
    out_s2 = self.norm2(out_s2)
    out_s2 = self.block2(out_s2)
    out = MEF.relu(out_s2)

    out_s4 = self.conv3(out)
    out_s4 = self.norm3(out_s4)
    out_s4 = self.block3(out_s4)
    out = MEF.relu(out_s4)

    out_s8 = self.conv4(out)
    out_s8 = self.norm4(out_s8)
    out_s8 = self.block4(out_s8)
    out_bottle = MEF.relu(out_s8)
    
    out = self.conv4_tr(out_bottle)
    out = self.norm4_tr(out)
    out = self.block4_tr(out)
    out_s4_tr = MEF.relu(out)                      

    out = ME.cat(out_s4_tr, out_s4)

    out = self.conv3_tr(out)
    out = self.norm3_tr(out)
    out = self.block3_tr(out)
    out_s2_tr = MEF.relu(out)

    out = ME.cat(out_s2_tr, out_s2)

    out = self.conv2_tr(out)
    out = self.norm2_tr(out)
    out = self.block2_tr(out)
    out_s1_tr = MEF.relu(out)

    out = ME.cat(out_s1_tr, out_s1)
    out_feat = self.conv1_tr(out)
    out_feat = MEF.relu(out_feat)
    out_feat = self.final(out_feat)

    out_att = self.conv1_tr_att(out_bottle)
    out_att = MEF.relu(out_att)

    out_att = self.conv2_tr_att(out_att)
    out_att = MEF.relu(out_att)
    out_att = self.glob_avg(out_att)

    out_att = self.final_att(out_att)
    out_att = ME.SparseTensor(self.final_att_act(out_att.F), 
        coordinate_map_key=out_att.coordinate_map_key, 
        coordinate_manager=out_att.coordinate_manager)
    
    out_feat = ME.SparseTensor(
      out_feat.F / torch.norm(out_feat.F, p=2, dim=1, keepdim=True),
      coordinate_map_key=out.coordinate_map_key,
      coordinate_manager=out.coordinate_manager)  
    
    return out_feat, out_att
      
    
def point_permute(p):
    return p.permute(1,0)
  
def xyz_permute(xyz):
    return xyz.permute(1,0)