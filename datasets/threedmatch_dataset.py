from math import degrees
import os 
import open3d as o3d
import numpy as np
import torch
import glob
import random
import logging

from scipy.spatial.transform import Rotation
from scipy.linalg import expm, norm
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset

from tools.utils import load_obj, to_tsfm, to_o3d_pcd, to_tensor, get_correspondences
from tools.model_util import npmat2euler, rotationMatrixToEulerAngles
from tools.file import read_trajectory
from datasets.basic_dataset import BasicDataset
from tools.pointcloud import get_matching_indices, make_open3d_point_cloud, draw_registration_result
from tools.transforms import apply_transform, decompose_rotation_translation

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


class ThreeDMatchPairDataset(BasicDataset):
  
    AUGMENT = None

    OVERLAP_RATIO = 0.3
    DATA_FILES = {
      'train': './datasets/split/indoor/train_3dmatch.txt',
      'val': './datasets/split/indoor/val_3dmatch.txt',
      'test': './config/test_3dmatch.txt'
    }
    def __init__(self,
               phase,
               transform=None,
               random_rotation=False,
               random_scale=False,
               manual_seed=False,
               config=None):
    #def __init__(self, infos, config):

        #super(ThreeDMatchPairDataset, self).__init__()
        #self.infos = infos
        self.root = root = config.threed_match_dir
        self.data_augmentation=True
        self.config = config
        self.voxel_size = config.voxel_size
        self.search_voxel_size = config.voxel_size * config.positive_pair_search_voxel_size_multiplier
        self.num_points = config.num_points
        self.rot_factor=1
        self.augment_noise = config.augment_noise

        BasicDataset.__init__(self, phase, transform, random_rotation, random_scale,
                         manual_seed, config)
        self.root = root = config.threed_match_dir
        logging.info(f"Loading the subset {phase} from {root}")

        subset_names = open(self.DATA_FILES[phase]).read().split()
        for name in subset_names:
            fname = name + "*%.2f.txt" % self.OVERLAP_RATIO
            fnames_txt = glob.glob(root + "/" + fname)
            assert len(fnames_txt) > 0, f"Make sure that the path {root} has data {fname}"
            for fname_txt in fnames_txt:
                with open(fname_txt) as f:
                    content = f.readlines()
                fnames = [x.strip().split() for x in content]
                for fname in fnames:
                    self.files.append([fname[0], fname[1]])
        
    def __len__(self):
        return len(self.files)

    """ def __len__(self):
        return len(self.infos['rot']) """
        
    def ground_truth_attention(self, p1, p2, trans):

        #draw_registration_result(p1, p2, trans)
        
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

    def __getitem__(self, idx):
        # get transformation
        
        file0 = os.path.join(self.root, self.files[idx][0])
        file1 = os.path.join(self.root, self.files[idx][1])
        data0 = np.load(file0)
        data1 = np.load(file1)

        src_pcd = data0["pcd"]
        tgt_pcd = data1["pcd"]
        color0 = data0["color"]
        color1 = data1["color"] 
        matching_search_voxel_size = self.matching_search_voxel_size

        if self.random_scale and random.random() < 0.95:
            scale = self.min_scale + (self.max_scale - self.min_scale) * random.random()
            matching_search_voxel_size *= scale
            src_pcd = scale * src_pcd
            tgt_pcd = scale * tgt_pcd

        if self.random_rotation:
            T0 = sample_random_trans(src_pcd, self.randg, self.rotation_range)
            T1 = sample_random_trans(tgt_pcd, self.randg, self.rotation_range)
            T_gt = (T1 @ np.linalg.inv(T0)).astype(np.float32)

            src_pcd = apply_transform(src_pcd, T0)
            tgt_pcd = apply_transform(tgt_pcd, T1)
        else:
            T_gt = np.identity(4)

        #Predator DataLoader
        """ # get transformation
        rot = self.infos['rot'][idx]
        trans = self.infos['trans'][idx]

        T_gt = to_tsfm(rot, trans)

        file0 = os.path.join(self.root, self.infos['src'][idx])
        file1 = os.path.join(self.root, self.infos['tgt'][idx])
        src_pcd = torch.load(file0)
        tgt_pcd = torch.load(file1) """

        search_voxel_size = self.search_voxel_size
        R, T = decompose_rotation_translation(T_gt)

        # add gaussian noise
        """ if self.data_augmentation:            
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
            tgt_pcd += (np.random.rand(tgt_pcd.shape[0],3) - 0.5) * self.augment_noise """


        euler = npmat2euler(R)

        pcd0 = make_open3d_point_cloud(src_pcd)
        pcd1 = make_open3d_point_cloud(tgt_pcd)

        # Voxelization
        _, sel0 = ME.utils.sparse_quantize(np.ascontiguousarray(src_pcd) / self.voxel_size, return_index=True)
        _, sel1 = ME.utils.sparse_quantize(np.ascontiguousarray(tgt_pcd) / self.voxel_size, return_index=True)

        src_xyz = src_pcd[sel0]
        tgt_xyz = tgt_pcd[sel1]

        # Select features and points using the returned voxelized indices
        pcd0.colors = o3d.utility.Vector3dVector(color0[sel0])
        pcd1.colors = o3d.utility.Vector3dVector(color1[sel1])
        pcd0.points = o3d.utility.Vector3dVector(np.array(pcd0.points)[sel0])
        pcd1.points = o3d.utility.Vector3dVector(np.array(pcd1.points)[sel1])
        # Get matches
        matching_inds = get_matching_indices(pcd0, pcd1, T_gt, matching_search_voxel_size)

        src_over, tgt_over, over_index0, over_index1 = self.ground_truth_attention(src_xyz, tgt_xyz, T_gt)

        # get voxelized coordinates
        src_coords, tgt_coords = np.floor(src_xyz / self.voxel_size), np.floor(tgt_xyz / self.voxel_size)

        npts0 = len(pcd0.colors)
        npts1 = len(pcd1.colors)

        feats_train0, feats_train1 = [], []

        # get feats
        feats_train0.append(np.ones((npts0,1),dtype=np.float32))
        feats_train1.append(np.ones((npts1,1),dtype=np.float32))

        feats0 = np.hstack(feats_train0)
        feats1 = np.hstack(feats_train1)

        over_matching_inds = get_matching_indices(to_o3d_pcd(src_over), to_o3d_pcd(tgt_over), T_gt, search_voxel_size)
        # overlap
        src_over_coords, tgt_over_coords = np.floor(src_over / self.voxel_size), np.floor(tgt_over / self.voxel_size) 

        # get over feats
        src_over_feats = np.ones((src_over_coords.shape[0],1),dtype=np.float32)
        tgt_over_feats = np.ones((tgt_over_coords.shape[0],1),dtype=np.float32)

        #src_xyz, tgt_xyz = to_tensor(src_xyz).float(), to_tensor(tgt_xyz).float()
        src_over, tgt_over = to_tensor(src_over).float(), to_tensor(tgt_over).float()
        over_index0, over_index1 = to_tensor(over_index0).int(), to_tensor(over_index1).int()
        
        scale = 1

        return (src_xyz, tgt_xyz, src_coords, tgt_coords, feats0, feats1 ,\
            src_over, tgt_over, src_over_coords, tgt_over_coords, src_over_feats, tgt_over_feats, \
            over_index0, over_index1, matching_inds, over_matching_inds, T_gt, euler, scale)