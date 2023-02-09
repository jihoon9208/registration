
import os 
import open3d as o3d
import numpy as np
import torch
import glob
import random
import logging

from scipy.linalg import expm, norm
from torch.utils.data import Dataset
from pytorch3d.transforms.rotation_conversions import matrix_to_euler_angles

from tools.utils import to_o3d_pcd, to_tensor
from tools.file import read_trajectory

from datasets.basic_dataset import BasicDataset
from tools.pointcloud import get_matching_indices, overlap_get_matching_indices ,make_open3d_point_cloud, ground_truth_attention
from tools.transforms import apply_transform, decompose_rotation_translation, sample_random_trans, M

import MinkowskiEngine as ME

class ThreeDMatchPairDataset(BasicDataset):
  
    AUGMENT = None

    OVERLAP_RATIO = 0.3
    DATA_FILES = {
      'train': './datasets/split/indoor/train_3dmatch.txt',
      'val': './datasets/split/indoor/val_3dmatch.txt',
      'test': './datasets/split/indoor/test_3dmatch.txt'
    }
    def __init__(self,
               phase,
               transform=None,
               random_rotation=True,
               random_scale=False,
               manual_seed=False,
               config=None):

        self.config = config
        self.root = root = config.threed_match_dir
        self.search_voxel_size = config.voxel_size * config.positive_pair_search_voxel_size_multiplier

        BasicDataset.__init__(self, phase, transform, random_rotation, random_scale,
                         manual_seed, config)
        
        logging.info(f"Loading the subset {phase} from {root}")

        subset_names = open(self.DATA_FILES[phase]).read().split()
        for name in subset_names:
            fname = name + "*%.2f.txt" % self.OVERLAP_RATIO
            fnames_txt = glob.glob(root + "/" + fname)
            if phase == 'train' or phase == 'val':
                assert len(fnames_txt) > 0, f"Make sure that the path {root} has data {fname}"
            for fname_txt in fnames_txt:
                with open(fname_txt) as f:
                    content = f.readlines()
                fnames = [x.strip().split() for x in content]
                for fname in fnames:
                    self.files.append([fname[0], fname[1]])
        
    def __len__(self):
        return len(self.files)
        
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
        matching_search_voxel_size = self.search_voxel_size

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

        R, T = decompose_rotation_translation(T_gt)

        euler = matrix_to_euler_angles(torch.from_numpy(R), "ZYX")

        # Voxelization
        _, sel0 = ME.utils.sparse_quantize(np.ascontiguousarray(src_pcd) / self.voxel_size, return_index=True)
        _, sel1 = ME.utils.sparse_quantize(np.ascontiguousarray(tgt_pcd) / self.voxel_size, return_index=True)

        src_xyz = src_pcd[sel0]
        tgt_xyz = tgt_pcd[sel1]

        pcd0 = make_open3d_point_cloud(src_pcd)
        pcd1 = make_open3d_point_cloud(tgt_pcd)

        # Select features and points using the returned voxelized indices
        pcd0.colors = o3d.utility.Vector3dVector(color0[sel0])
        pcd1.colors = o3d.utility.Vector3dVector(color1[sel1])
        pcd0.points = o3d.utility.Vector3dVector(np.array(pcd0.points)[sel0])
        pcd1.points = o3d.utility.Vector3dVector(np.array(pcd1.points)[sel1])

        # Get matches
        matching_inds = get_matching_indices(pcd0, pcd1, T_gt, matching_search_voxel_size)
        src_over, tgt_over, over_index0, over_index1 = ground_truth_attention(src_xyz, tgt_xyz, T_gt, matching_search_voxel_size)      

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

        over_matching_inds = overlap_get_matching_indices(to_o3d_pcd(src_over), to_o3d_pcd(tgt_over), T_gt)
        
        src_over, tgt_over = to_tensor(src_over).float(), to_tensor(tgt_over).float()
        over_index0, over_index1 = to_tensor(over_index0).int(), to_tensor(over_index1).int()

        scale = 1

        return (src_xyz, tgt_xyz, src_coords, tgt_coords, feats0, feats1, src_over, tgt_over,\
            over_index0, over_index1, matching_inds, over_matching_inds, T_gt, euler, scale)

class ThreeDMatchTestDataset(Dataset):

    DATA_FILES = {"test": "./datasets/split/indoor/test_3dmatch.txt"}
    CONFIG_ROOT = "./datasets/config/3DMatch"

    TE_THRESH = 30
    RE_THRESH = 15 

    def __init__(self, root):

        self.root = root
        logging.info(f"Loading the subset test from {root}")
        subset_names = open(self.DATA_FILES["test"]).read().split()
        self.subset_names = subset_names

        self.files = []
        for sname in subset_names:
            traj_file = os.path.join(self.CONFIG_ROOT, sname, "gt.log")
            assert os.path.exists(traj_file)
            traj = read_trajectory(traj_file)
            for ctraj in traj:
                i = ctraj.metadata[0]
                j = ctraj.metadata[1]
                T_gt = ctraj.pose
                self.files.append((sname, i, j, T_gt))
        logging.info(f"Loaded {self.__class__.__name__} with {len(self.files)} data")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        
        sname, i, j, T_gt = self.files[idx]
        file0 = os.path.join(self.root, sname, f"cloud_bin_{i}.ply")
        file1 = os.path.join(self.root, sname, f"cloud_bin_{j}.ply")

        pcd0 = o3d.io.read_point_cloud(file0)
        pcd1 = o3d.io.read_point_cloud(file1)

        xyz0 = np.asarray(pcd0.points).astype(np.float32)
        xyz1 = np.asarray(pcd1.points).astype(np.float32)

        return sname, xyz0, xyz1, T_gt, file0, file1

class ThreeDLoMatchTestDataset(ThreeDMatchTestDataset):
    """3DLoMatch test dataset"""

    SPLIT_FILES = {"test": "./datasets/splits/test_3dmatch.txt"}
    CONFIG_ROOT = "./datasets/config/3DLoMatch"