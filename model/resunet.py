import sys
from turtle import forward
from matplotlib.transforms import Transform
import torch
import math
import numpy as np
from typing import Any
import torch.nn as nn
import torch.nn.functional as F

import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MEF


from model.common import get_norm
from model.residual_block import get_block


def point_permute(p):
    return p.permute(1,0)
  
def xyz_permute(xyz):
    return xyz.permute(1,0)

class SparseResNetFull(ME.MinkowskiNetwork):

    NORM_TYPE = 'BN'
    BLOCK_NORM_TYPE = 'BN'
    CHANNELS = [None, 32, 64, 128, 256, 512]
    TR_CHANNELS = [None, 32, 64, 64, 128]

    def __init__(
                    self,
                    config,
                    in_channels=3,
                    out_channels = None,
                    bn_momentum=0.1,
                    conv1_kernel_size=None,
                    D=3):
        super().__init__(D=3)


        self.voxel_size = config.voxel_size
        self.batch_size = config.batch_size
        NORM_TYPE = self.NORM_TYPE
        BLOCK_NORM_TYPE = self.BLOCK_NORM_TYPE
        CHANNELS = self.CHANNELS
        TR_CHANNELS = self.TR_CHANNELS

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
            in_channels=CHANNELS[1] + TR_CHANNELS[2],
            out_channels=TR_CHANNELS[1],
            kernel_size=1,
            stride=1,
            dilation=1,
            bias=False,
            dimension=D)
        
        self.final = ME.MinkowskiConvolution(
            in_channels=TR_CHANNELS[1],
            out_channels=out_channels,
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

        self.final_att_act = torch.nn.Softplus(beta=1, threshold=20)
            
        ################
        # attention
        
    def forward(self, source):

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

        s_out = self.conv4_tr(s_out)
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
        source = self.conv1_tr(s_out)
        source = MEF.relu(source)
        source = self.final(source)
        
        out_att = self.conv1_tr_att(s_out)
        out_att = MEF.relu(out_att)
        out_att = self.final_att(out_att)
        out_att = ME.SparseTensor(self.final_att_act(out_att.F), 
            coordinate_map_key=out_att.coordinate_map_key, 
            coordinate_manager=out_att.coordinate_manager)
        
        source = ME.SparseTensor(
        source.F / torch.norm(source.F, p=2, dim=1, keepdim=True),
        coordinate_map_key=source.coordinate_map_key,
        coordinate_manager=source.coordinate_manager)  
        
        return source, out_att


class SparseResNetOver(ME.MinkowskiNetwork):

    NORM_TYPE = 'BN'
    BLOCK_NORM_TYPE = 'BN'
    CHANNELS = [None, 32, 64, 128, 256, 512]
    TR_CHANNELS = [None, 32, 64, 64, 128]

    def __init__(
                    self,
                    config,
                    in_channels=3,
                    out_channels = None,
                    bn_momentum=0.1,
                    conv1_kernel_size=None,
                    D=3):
        super().__init__(D=3)

        self.voxel_size = config.voxel_size
        self.batch_size = config.batch_size
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

        self.conv2_tr = ME.MinkowskiConvolutionTranspose(
            
            in_channels=CHANNELS[2] ,        
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
        
    def forward(self, source):

        # source feature 
        source_s1 = self.conv1(source)
        source_s1 = self.norm1(source_s1)
        source_s1 = self.block1(source_s1)
        s_out = MEF.relu(source_s1)

        source_s2 = self.conv2(s_out)
        source_s2 = self.norm2(source_s2)
        source_s2 = self.block2(source_s2)
        s_out = MEF.relu(source_s2)


        s_out = self.conv2_tr(s_out)
        s_out = self.norm2_tr(s_out)
        s_out = self.block2_tr(s_out)
        source_s1_tr = MEF.relu(s_out)

        s_out = ME.cat(source_s1_tr, source_s1)
        s_out = self.conv1_tr(s_out)
        source = MEF.relu(s_out)
        
        source = ME.SparseTensor(
            source.F / torch.norm(source.F, p=2, dim=1, keepdim=True),
            coordinate_map_key=source.coordinate_map_key,
            coordinate_manager=source.coordinate_manager)  

        return source.F