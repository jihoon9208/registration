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

def ground_truth_attention( p1, p2, trans, overlap_radius):
    
    ideal_pts2 = apply_transform(p1, trans) 

    #ind = np.random.permutation(int(round(self.num_points/10)))

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

def overlap_get_matching_indices(source, target, trans, K=None):
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

def draw_registration_result(source, target, transformation):
    #source_temp = copy.deepcopy(source)
    #target_temp = copy.deepcopy(target)

    source_temp = make_open3d_point_cloud(source)
    target_temp = make_open3d_point_cloud(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

def draw_point_cloud(source, target, transformation):

    source_temp = make_open3d_point_cloud(source)
    target_temp = make_open3d_point_cloud(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp])
    o3d.visualization.draw_geometries([target_temp])