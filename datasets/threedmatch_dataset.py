import os 
import open3d as o3d
import numpy as np
import torch

from scipy.spatial.transform import Rotation
from scipy.linalg import expm, norm
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset

from tools.utils import load_obj, to_tsfm, to_o3d_pcd, to_tensor, get_correspondences
from tools.model_util import npmat2euler
from tools.file import read_trajectory
from tools.pointcloud import get_matching_indices, make_open3d_point_cloud
from tools.transforms import sample_points

import MinkowskiEngine as ME

# Rotation matrix along axis with angle theta
def M(axis, theta):
    return expm(np.cross(np.eye(3), axis / norm(axis) * theta))

def sample_random_trans(pcd, randg, rotation_range=360):
    T = np.eye(4)
    R = M(randg.rand(3) - 0.5, rotation_range * np.pi / 180.0 * (randg.rand(1) - 0.5))
    T[:3, :3] = R
    T[:3, 3] = R.dot(-np.mean(pcd, axis=0))
    return T


class ThreeDMatchPairDataset(Dataset):
  
    AUGMENT = None

    OVERLAP_RATIO = 0.3


    def __init__(self, infos, config, data_augmentation=True):
        super(ThreeDMatchPairDataset,self).__init__()

        self.infos = infos
        self.root = root = config.threed_match_dir
        self.data_augmentation=data_augmentation
        self.config = config
        self.voxel_size = config.voxel_size
        self.search_voxel_size = config.voxel_size * config.positive_pair_search_voxel_size_multiplier
        self.num_points = config.num_points
        self.rot_factor=1.
        self.augment_noise = config.augment_noise
        
    def __len__(self):
        return len(self.infos['rot'])
    
    def apply_transform(self, pts, trans):
        R = trans[:3, :3]
        T = trans[:3, 3]
        return pts @ R.T + T
    
    def ground_truth_attention(self, p1, p2, trans):
        
        p1 = sample_points(p1, self.num_points)
        p2 = sample_points(p2, self.num_points)

        ideal_pts2 = self.apply_transform(p1, trans)    

        nn = NearestNeighbors(n_neighbors=1).fit(p2)
        distance, neighbors = nn.kneighbors(ideal_pts2)
        neighbors1 = neighbors[distance < 0.3]
        pcd1 = p2[neighbors1]
        
        # Search NN for each p2 in ideal_pt2
        nn = NearestNeighbors(n_neighbors=1).fit(ideal_pts2)
        distance, neighbors = nn.kneighbors(p2)
        neighbors2 = neighbors[distance < 0.3]
        pcd0 = p1[neighbors2]
    
        return pcd0, pcd1, neighbors2, neighbors1

    def __getitem__(self, idx):
        # get transformation
        rot = self.infos['rot'][idx]
        trans = self.infos['trans'][idx]

        tsfm = to_tsfm(rot, trans)

        file0 = os.path.join(self.root, self.infos['src'][idx])
        file1 = os.path.join(self.root, self.infos['tgt'][idx])
        src_pcd = torch.load(file0)
        tgt_pcd = torch.load(file1)

        search_voxel_size = self.search_voxel_size

        # add gaussian noise
        if self.data_augmentation:            
            # rotate the point cloud
            euler_ab = np.random.rand(3) * np.pi * 2 / self.rot_factor # anglez, angley, anglex
            rot_ab = Rotation.from_euler('zyx', euler_ab).as_matrix()
            if(np.random.rand(1)[0] > 0.5):
                src_pcd = np.matmul(rot_ab,src_pcd.T).T
                rot = np.matmul(rot,rot_ab.T)
            else:
                tgt_pcd = np.matmul(rot_ab,tgt_pcd.T).T
                rot = np.matmul(rot_ab,rot)
                trans = np.matmul(rot_ab,trans)

            src_pcd += (np.random.rand(src_pcd.shape[0],3) - 0.5) * self.augment_noise
            tgt_pcd += (np.random.rand(tgt_pcd.shape[0],3) - 0.5) * self.augment_noise
        

        """ T0 = sample_random_trans(xyz0, self.randg, self.rotation_range)
        T1 = sample_random_trans(xyz1, self.randg, self.rotation_range)
        trans = T1 @ np.linalg.inv(T0)

        xyz0 = self.apply_transform(xyz0, T0)
        xyz1 = self.apply_transform(xyz1, T1)
        """
        
        euler = npmat2euler(rot, 'zyx')

        # Voxelization
        _, sel0 = ME.utils.sparse_quantize(np.ascontiguousarray(src_pcd) / self.voxel_size, return_index=True)
        _, sel1 = ME.utils.sparse_quantize(np.ascontiguousarray(tgt_pcd) / self.voxel_size, return_index=True)

        # get correspondence
        tsfm = to_tsfm(rot, trans)
        src_xyz, tgt_xyz = src_pcd[sel0], tgt_pcd[sel1] # raw point clouds
        matching_inds = get_correspondences(to_o3d_pcd(src_xyz), to_o3d_pcd(tgt_xyz), tsfm, search_voxel_size)

        src_over, tgt_over, over_index0, over_index1 = self.ground_truth_attention(src_xyz, tgt_xyz, tsfm)

        # get voxelized coordinates
        src_coords, tgt_coords = np.floor(src_xyz / self.voxel_size), np.floor(tgt_xyz / self.voxel_size)

        # get feats
        src_feats = np.ones((src_coords.shape[0],1),dtype=np.float32)
        tgt_feats = np.ones((tgt_coords.shape[0],1),dtype=np.float32)

        src_xyz, tgt_xyz = to_tensor(src_xyz).float(), to_tensor(tgt_xyz).float()
        src_over, tgt_over = to_tensor(src_over).float(), to_tensor(tgt_over).float()
        over_index0, over_index1 = to_tensor(over_index0).int(), to_tensor(over_index1).int()
        rot, trans = to_tensor(rot), to_tensor(trans)
        scale = 1

        return (src_xyz, tgt_xyz, src_coords, tgt_coords, src_feats, tgt_feats, src_over, tgt_over, \
            over_index0, over_index1, matching_inds, rot, trans, euler, scale)