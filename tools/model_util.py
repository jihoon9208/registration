# -*- coding: future_fstrings -*-
from __future__ import print_function
import torch
import numpy as np
from scipy.spatial.transform import Rotation
import torch.nn.functional as F

# Part of the code is referred from: https://github.com/ClementPinard/SfmLearner-Pytorch/blob/master/inverse_warp.py
# Part of the code is referred from: https://github.com/WangYueFt/dcp


def transform_point_cloud(point_cloud, rotation, translation):
    
    return point_cloud @ rotation.T + translation

def transform_point_cloud0(point_cloud, rotation, translation):
    
    rot_mat = rotation
    return point_cloud @ rot_mat.T + translation.T

def npmat2euler(mats, seq='zyx'):
    eulers = []
    for i in range(mats.shape[0]):
        r = Rotation.from_mrp(mats[i].T)
        eulers.append(r.as_euler(seq, degrees=True))
    return np.asarray(eulers, dtype='float32')

def square_dist_torch(src, dst):

    N, _ = src.shape
    M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(1,0))     
    dist += torch.sum(src ** 2, -1).view(N, 1)
    dist += torch.sum(dst ** 2, -1).view(1, M)

    return dist 

def new_cdist(x1, x2):
    x1 = x1.float()
    x2 = x2.float()
    #x1_norm = x1.pow(2).sum(dim=-1, keepdim=True).float()
    #x2_norm = x2.pow(2).sum(dim=-1, keepdim=True).float()
    res = torch.matmul(x1, x2.T)
    res = res.clamp_min_(1e-30).sqrt_()
    return res

def dist_torch(A,B):
    """
    Measure Squared Euclidean Distance from every point in point-cloud A, to every point in point-cloud B
    :param A: Point Cloud: Nx3 Array of real numbers, each row represents one point in x,y,z space
    :param B: Point Cloud: Mx3 Array of real numbers
    :return:  NxM array, where element [i,j] is the squared distance between the i'th point in A and the j'th point in B
    """

    s = torch.matmul(A.float(), B.float().T)
    s[s<0]=0
    return torch.sqrt(s)

def dist_feat(A, B):

    feat_map = torch.matmul(A.float().T, B.float())
    feat_map[feat_map<0]=0
    return torch.sqrt(feat_map)

def cdist_torch(A, B, points_dim=None):
    num_features = 64
    if points_dim is not None:
        num_features = points_dim
    if (A.shape[-1] != num_features):
        A = torch.transpose(A, dim0=-2, dim1=-1)
    if (B.shape[-1] != num_features):
        B = torch.transpose(B, dim0=-2, dim1=-1)
    assert A.shape[-1] == num_features
    assert B.shape[-1] == num_features
    A = A.double().contiguous()
    B = B.double().contiguous()
    C = new_cdist(A,B)
    return C

def min_without_self_per_row_torch(D):
    """
    Accepts a distance matrix between all points in a set. For each point,
    returns its distance from the closest point that is not itself.
    :param D: Distance matrix, where element [i,j] is the distance between i'th point in the set and the j'th point in the set. Should be symmetric with zeros on the diagonal.
    :return: vector of distances to nearest neighbor for each point.
    """
    E = D.clone()
    diag_ind = range(E.shape[0])
    E[diag_ind,diag_ind] = np.inf
    m = E.min(dim=1).values
    return m

def representative_neighbor_dist_torch(D):
    """
    Accepts a distance matrix between all points in a set,
    returns a number that is representative of the distances in this set.

    :param D: Distance matrix, where element [i,j] is the distance between i'th point in the set and the j'th point in the set. Should be symmetric with zeros on the diagonal.
    :return: The representative distance in this set
    """

    assert D.shape[0] == D.shape[1], "Input to representative_neighbor_dist should be a matrix of distances from a point cloud to itself"
    m = min_without_self_per_row_torch(D)
    neighbor_dist = m.median()
    return neighbor_dist.cpu().detach().numpy()

def guess_best_alpha_torch1(A,dim_num=3, transpose=None):
    """
        A good guess for the temperature of the soft argmin (alpha) can
        be calculated as a linear function of the representative (e.g. median)
        distance of points to their nearest neighbor in a point cloud.

        :param A: Point Cloud of size Nx3
        :return: Estimated value of alpha
        """

    COEFF = 0.1
    EPS = 1e-8
    if transpose is None:
        assert A.shape[0] != A.shape[1], 'Number of points and number of dimensions can''t be same'
    if (A.shape[1] != dim_num and transpose is None) or transpose:
        A = A.T
    assert A.shape[1]==dim_num
    rep = representative_neighbor_dist_torch(dist_torch(A, A))
    return COEFF * rep + EPS

def guess_best_alpha_torch( feat, dim_num=3, transpose=None):
    
    COEFF = 0.1
    EPS = 1e-8
    
    feat_map = dist_feat(feat, feat)
    diag_ind = range(feat_map.shape[0])
    feat_map[diag_ind,diag_ind] = np.inf
    feat_min = feat_map.min(dim=1).values
    neighbor_dist = feat_min.mean()

    return COEFF * neighbor_dist + EPS



def soft_BBS_loss_torch(S, T, temperature, points_dim=None, return_mat=False, transpose=None):
    num_features = 64

    T_num_samples = T.shape[1]
    S_num_samples = S.shape[1]

    mean_num_samples = np.mean([S_num_samples, T_num_samples])

    D = cdist_torch( S, T, points_dim)
    R = torch.squeeze(softargmin_rows_torch(D, temperature))
    C = torch.squeeze(softargmin_rows_torch(D.T, temperature))
    B = torch.mul(R, C.T)
    loss = torch.div(-torch.sum(B), mean_num_samples).view(1)
    if return_mat:
        return B
    else:
        return loss

def my_softmax(x, eps=1e-12):
    x_exp = torch.exp(x - x.min())
    x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True)
    return x_exp/(x_exp_sum + eps)

def softargmin_rows_torch(X, t, eps=1e-12):
    t = t.double()
    X = X.double()
    weights = my_softmax(-X/t, eps=eps)
    return weights
