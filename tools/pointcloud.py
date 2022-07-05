# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu) and Wei Dong (weidong@andrew.cmu.edu)
#
# Please cite the following papers if you use any part of the code.
# - Christopher Choy, Wei Dong, Vladlen Koltun, Deep Global Registration, CVPR 2020
# - Christopher Choy, Jaesik Park, Vladlen Koltun, Fully Convolutional Geometric Features, ICCV 2019
# - Christopher Choy, JunYoung Gwak, Silvio Savarese, 4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks, CVPR 2019
import copy
import numpy as np
import torch
import math
from sklearn.neighbors import NearestNeighbors
from tools.transforms import apply_transform 
import open3d as o3d
from tools.utils import to_o3d_pcd


def make_open3d_point_cloud(xyz, color=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if color is not None:
        if len(color) != len(xyz):
            color = np.tile(color, (len(xyz), 1))
        pcd.colors = o3d.utility.Vector3dVector(color)
    return pcd


def make_open3d_feature(data, dim, npts):
    feature = o3d.registration.Feature()
    feature.resize(dim, npts)
    feature.data = data.cpu().numpy().astype('d').transpose()
    return feature


def make_open3d_feature_from_numpy(data):
    assert isinstance(data, np.ndarray)
    assert data.ndim == 2

    feature = o3d.registration.Feature()
    feature.resize(data.shape[1], data.shape[0])
    feature.data = data.astype('d').transpose()
    return feature


def pointcloud_to_spheres(pcd, voxel_size, color, sphere_size=0.6):
    spheres = o3d.geometry.TriangleMesh()
    s = o3d.geometry.TriangleMesh.create_sphere(radius=voxel_size * sphere_size)
    s.compute_vertex_normals()
    s.paint_uniform_color(color)
    if isinstance(pcd, o3d.geometry.PointCloud):
        pcd = np.array(pcd.points)
    for i, p in enumerate(pcd):
        si = copy.deepcopy(s)
        trans = np.identity(4)
        trans[:3, 3] = p
        si.transform(trans)
        # si.paint_uniform_color(pcd.colors[i])
        spheres += si
    return spheres


def prepare_single_pointcloud(pcd, voxel_size):
    pcd.estimate_normals(o3d.KDTreeSearchParamHybrid(radius=voxel_size * 2.0, max_nn=30))
    return pcd


def prepare_pointcloud(filename, voxel_size):
    pcd = o3d.io.read_point_cloud(filename)
    T = get_random_transformation(pcd)
    pcd.transform(T)
    pcd_down = pcd.voxel_down_sample(voxel_size)
    return pcd_down, T


def compute_overlap_ratio(pcd0, pcd1, trans, voxel_size):
    pcd0_down = pcd0.voxel_down_sample(voxel_size)
    pcd1_down = pcd1.voxel_down_sample(voxel_size)
    matching01 = get_matching_indices(pcd0_down, pcd1_down, trans, voxel_size, 1)
    matching10 = get_matching_indices(pcd1_down, pcd0_down, np.linalg.inv(trans),
                                        voxel_size, 1)
    overlap0 = len(matching01) / len(pcd0_down.points)
    overlap1 = len(matching10) / len(pcd1_down.points)
    return max(overlap0, overlap1)


def get_matching_indices(source, target, trans, search_voxel_size, K=None):
    source_copy = copy.deepcopy(source)
    target_copy = copy.deepcopy(target)
    source_copy.transform(trans)
    pcd_tree = o3d.geometry.KDTreeFlann(target_copy)

    match_inds = []
    for i, point in enumerate(source_copy.points):
        [_, idx, _] = pcd_tree.search_radius_vector_3d(point, search_voxel_size)
        if K is not None:
            idx = idx[:K]
        for j in idx:
            match_inds.append((i, j))
    return match_inds

def ground_truth_attention(p1, p2, trans, overlap_radius):
        
    ideal_pts2 = apply_transform(p1, trans) 

    nn = NearestNeighbors(n_neighbors=1).fit(p2)
    distance, neighbors1 = nn.kneighbors(ideal_pts2)
    neighbors1 = neighbors1[distance < overlap_radius * 2]
    
    # Search NN for each p2 in ideal_pt2
    nn = NearestNeighbors(n_neighbors=1).fit(ideal_pts2)
    distance, neighbors2 = nn.kneighbors(p2)
    neighbors2 = neighbors2[distance < overlap_radius * 2]

    N = min(len(neighbors1), len(neighbors2))
    ind = np.random.permutation(N)

    pcd1 = p2[neighbors1[ind]]
    pcd0 = p1[neighbors2[ind]]

    return pcd0, pcd1, neighbors2[ind], neighbors1[ind]


def compute_overlap(p1, p2, overlap_radius):

    if isinstance(p1, np.ndarray):
        src_pcd = to_o3d_pcd(p1)
        src_xyz = p1
    else:
        src_pcd = p1
        src_xyz = np.asarray(p1)

    if isinstance(p2, np.ndarray):
        tgt_pcd = to_o3d_pcd(p2)
        tgt_xyz = p2
    else:
        tgt_pcd = p2
        tgt_xyz = np.asarray(p2)

    # Check which points in tgt has a correspondence (i.e. point nearby) in the src,
    # and then in the other direction. As long there's a point nearby, it's
    # considered to be in the overlap region. For correspondences, we require a stronger
    # condition of being mutual matches
    tgt_corr = np.full(tgt_xyz.shape[0], -1)
    pcd_tree = o3d.geometry.KDTreeFlann(src_pcd)
    for i, t in enumerate(tgt_xyz):
        num_knn, knn_indices, knn_dist = pcd_tree.search_radius_vector_3d(t, overlap_radius)
        if num_knn > 0:
            tgt_corr[i] = knn_indices[0]

    src_corr = np.full(src_xyz.shape[0], -1)    
    pcd_tree = o3d.geometry.KDTreeFlann(tgt_pcd)
    for i, s in enumerate(src_xyz):
        num_knn, knn_indices, knn_dist = pcd_tree.search_radius_vector_3d(s, overlap_radius)
        if num_knn > 0:
            src_corr[i] = knn_indices[0]

    # Compute mutual correspondences
    src_corr_is_mutual = np.logical_and(tgt_corr[src_corr] == np.arange(len(src_corr)),
                                        src_corr > 0)
    tgt_corr_is_mutual = np.logical_and(src_corr[tgt_corr] == np.arange(len(tgt_corr)),
                                        tgt_corr > 0)

    src_tgt_corr = np.stack([np.nonzero(src_corr_is_mutual)[0],
                             src_corr[src_corr_is_mutual]])
    tgt_src_corr = np.stack([np.nonzero(tgt_corr_is_mutual)[0],
                             tgt_corr[tgt_corr_is_mutual]])

    src = torch.from_numpy(src_xyz[src_tgt_corr[1]])
    tgt = torch.from_numpy(tgt_xyz[tgt_src_corr[1]])

    return src, tgt, src_tgt_corr[1], tgt_src_corr[1]

def overlap_get_matching_indices(source, target, trans, search_voxel_size, K=None):
    source_copy = copy.deepcopy(source)
    target_copy = copy.deepcopy(target)
    source_copy.transform(trans)
    pcd_tree = o3d.geometry.KDTreeFlann(target_copy)

    match_inds = []
    for i, point in enumerate(source_copy.points):
        [_, idx, _] = pcd_tree.search_knn_vector_3d(point, 1)
        if K is not None:
            idx = idx[:K]
        for j in idx:
            match_inds.append((i, j))
    return match_inds


def evaluate_feature(pcd0, pcd1, feat0, feat1, trans_gth, search_voxel_size):
    match_inds = get_matching_indices(pcd0, pcd1, trans_gth, search_voxel_size)
    pcd_tree = o3d.geometry.KDTreeFlann(feat1)
    dist = []
    for ind in match_inds:
        k, idx, _ = pcd_tree.search_knn_vector_xd(feat0.data[:, ind[0]], 1)
        dist.append(
            np.clip(np.power(pcd1.points[ind[1]] - pcd1.points[idx[0]], 2),
                    a_min=0.0,
                    a_max=1.0))
    return np.mean(dist)


def evaluate_feature_3dmatch(pcd0, pcd1, feat0, feat1, trans_gth, inlier_thresh=0.1):
    r"""Return the hit ratio (ratio of inlier correspondences and all correspondences).

    inliear_thresh is the inlier_threshold in meter.
    """
    if len(pcd0.points) < len(pcd1.points):
        hit = valid_feat_ratio(pcd0, pcd1, feat0, feat1, trans_gth, inlier_thresh)
    else:
        hit = valid_feat_ratio(pcd1, pcd0, feat1, feat0, np.linalg.inv(trans_gth),
                            inlier_thresh)
    return hit


def get_matching_matrix(source, target, trans, voxel_size, debug_mode):
    source_copy = copy.deepcopy(source)
    target_copy = copy.deepcopy(target)
    source_copy.transform(trans)
    pcd_tree = o3d.geometry.KDTreeFlann(target_copy)
    matching_matrix = np.zeros((len(source_copy.points), len(target_copy.points)))

    for i, point in enumerate(source_copy.points):
        [k, idx, _] = pcd_tree.search_radius_vector_3d(point, voxel_size * 1.5)
        if k >= 1:
            matching_matrix[i, idx[0]] = 1  # TODO: only the cloest?

    return matching_matrix


def get_random_transformation(pcd_input):
  def rot_x(x):
    out = np.zeros((3, 3))
    c = math.cos(x)
    s = math.sin(x)
    out[0, 0] = 1
    out[1, 1] = c
    out[1, 2] = -s
    out[2, 1] = s
    out[2, 2] = c
    return out

  def rot_y(x):
    out = np.zeros((3, 3))
    c = math.cos(x)
    s = math.sin(x)
    out[0, 0] = c
    out[0, 2] = s
    out[1, 1] = 1
    out[2, 0] = -s
    out[2, 2] = c
    return out

  def rot_z(x):
    out = np.zeros((3, 3))
    c = math.cos(x)
    s = math.sin(x)
    out[0, 0] = c
    out[0, 1] = -s
    out[1, 0] = s
    out[1, 1] = c
    out[2, 2] = 1
    return out

  pcd_output = copy.deepcopy(pcd_input)
  mean = np.mean(np.asarray(pcd_output.points), axis=0).transpose()
  xyz = np.random.uniform(0, 2 * math.pi, 3)
  R = np.dot(np.dot(rot_x(xyz[0]), rot_y(xyz[1])), rot_z(xyz[2]))
  T = np.zeros((4, 4))
  T[:3, :3] = R
  T[:3, 3] = np.dot(-R, mean)
  T[3, 3] = 1
  return T


def draw_registration_result(source, target, transformation):
    #source_temp = copy.deepcopy(source)
    #target_temp = copy.deepcopy(target)

    source_temp = make_open3d_point_cloud(source)
    target_temp = make_open3d_point_cloud(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])
