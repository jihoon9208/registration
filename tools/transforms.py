import torch
import numpy as np
import random
from scipy.linalg import expm, norm
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

#import open3d as o3d

def decompose_rotation_translation(Ts):
    
    R = Ts[:3, :3]
    T = Ts[:3, 3]

    return R, T

def ground_truth_attention( p1, p2, trans):
    
    ideal_pts2 = apply_transform(p1, trans) 

    #ind = np.random.permutation(int(round(self.num_points/10)))

    nn = NearestNeighbors(n_neighbors=1).fit(p2)
    distance, neighbors1 = nn.kneighbors(ideal_pts2)
    neighbors1 = neighbors1[distance < 0.3]
    
    # Search NN for each p2 in ideal_pt2
    nn = NearestNeighbors(n_neighbors=1).fit(ideal_pts2)
    distance, neighbors2 = nn.kneighbors(p2)
    neighbors2 = neighbors2[distance < 0.3]

    N = min(len(neighbors1), len(neighbors2))
    ind = np.random.permutation(N)

    pcd1 = p2[neighbors1[ind]]
    pcd0 = p1[neighbors2[ind]]

    return pcd0, pcd1, neighbors2[ind], neighbors1[ind]

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
