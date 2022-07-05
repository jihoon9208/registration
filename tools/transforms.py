# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu) and Wei Dong (weidong@andrew.cmu.edu)
#
# Please cite the following papers if you use any part of the code.
# - Christopher Choy, Wei Dong, Vladlen Koltun, Deep Global Registration, CVPR 2020
# - Christopher Choy, Jaesik Park, Vladlen Koltun, Fully Convolutional Geometric Features, ICCV 2019
# - Christopher Choy, JunYoung Gwak, Silvio Savarese, 4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks, CVPR 2019
import torch
import numpy as np
import random
from scipy.linalg import expm, norm
from sklearn.neighbors import NearestNeighbors


import open3d as o3d

def decompose_rotation_translation(Ts):
    
    R = Ts[:3, :3]
    T = Ts[:3, 3]

    return R, T

def voxelize(point_cloud, voxel_size):
    # Random permutation (for random selection within voxel)
    point_cloud = np.random.permutation(point_cloud)

    # Set minimum value to 0 on each axis
    min_val = point_cloud.min(0)
    pc = point_cloud - min_val

    # Quantize
    pc = np.floor(pc / voxel_size)
    L, M, N = pc.max(0) + 1
    pc = pc[:, 0] + L * pc[:, 1] + L * M * pc[:, 2]

    # Select voxel
    _, idx = np.unique(pc, return_index=True)

    return point_cloud[idx, :]

def sample_points(pts, num_points):
    if pts.shape[0] > num_points:
        pts = np.random.permutation(pts)[:num_points]
    else:
        pts = np.random.permutation(pts)
    return pts

# Rotation matrix along axis with angle theta
def M(axis, theta):
    return expm(np.cross(np.eye(3), axis / norm(axis) * theta))

def M_z_axis(theta):
    theta = theta * np.pi / 180.0
    z_axis = np.array([0, 0, 1])
    return expm(np.cross(np.eye(3), z_axis / norm(z_axis) * theta))

def sample_random_rotation_z_axis(pcd, randg, rotation_range=360):
    T = np.eye(4)
    z_axis = np.array([0, 0, 1])    
    random_angle = rotation_range * (randg.rand(1) - 0.5)
    R = M(z_axis, random_angle * np.pi / 180.0)
    T[:3, :3] = R
    T[:3, 3] = R.dot(-np.mean(pcd, axis=0))
    return T

def sample_random_trans(pcd, randg, rotation_range=360):
    T = np.eye(4)
    R = M(randg.rand(3) - 0.5, rotation_range * np.pi / 180.0 * (randg.rand(1) - 0.5))
    T[:3, :3] = R  
    T[:3, 3] = R.dot(-np.mean(pcd, axis=0))
    return T

def apply_transform(pts, trans):
    R = trans[:3, :3]
    T = trans[:3, 3]
    pts = pts @ R.T + T
    return pts

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, coords, feats):
        for transform in self.transforms:
            coords, feats = transform(coords, feats)
        return coords, feats

class Jitter:

    def __init__(self, mu=0, sigma=0.01):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, coords, feats):
        if random.random() < 0.95:
            if isinstance(feats, np.ndarray):
                feats += np.random.normal(self.mu, self.sigma, (feats.shape[0], feats.shape[1]))
            else:
                feats += (torch.randn_like(feats) * self.sigma) + self.mu
        return coords, feats

class ChromaticShift:
    def __init__(self, mu=0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, coords, feats):
        if random.random() < 0.95:
            feats[:, :3] += torch.randn(self.mu, self.sigma, (1, 3))
        return coords, feats
