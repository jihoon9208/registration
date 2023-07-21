# Basic libs
import os, glob, random, copy, torch
import numpy as np
import open3d
from scipy.spatial.transform import Rotation
from pytorch3d.transforms.rotation_conversions import matrix_to_euler_angles

# Dataset parent class
from torch.utils.data import Dataset
from tools.utils import to_o3d_pcd, to_tensor
from tools.pointcloud import draw_registration_result, get_matching_indices, overlap_get_matching_indices ,make_open3d_point_cloud, ground_truth_attention, compute_overlap
from tools.transforms import apply_transform, decompose_rotation_translation

import MinkowskiEngine as ME

class KITTIDataset(Dataset):
    """
    We follow D3Feat to add data augmentation part.
    We first voxelize the pcd and get matches
    Then we apply data augmentation to pcds. KPConv runs over processed pcds, but later for loss computation, we use pcds before data augmentation
    """
    DATA_FILES = {
        'train': './datasets/split/kitti/train_kitti.txt',
        'val': './datasets/split/kitti/val_kitti.txt',
        'test': './datasets/split/kitti/test_kitti.txt'
    }
    MIN_DIST = 10
    TE_THRESH = 60
    RE_THRESH = 5

    def __init__(self,
                split,
                transform=None,
                random_rotation=False,
                random_scale=False,
                manual_seed=False, 
                config= None
        ):
        super(KITTIDataset, self).__init__()
        self.config = config
        self.root = root = config.kitti_root + '/dataset'
        self.icp_path = os.path.join(config.kitti_root, 'icp')
        if not os.path.exists(self.icp_path):
            os.makedirs(self.icp_path)
        self.voxel_size = config.voxel_size
        self.matching_search_voxel_size = config.overlap_radius
        self.data_augmentation = config.data_augmentation
        self.augment_noise = config.augment_noise
        self.IS_ODOMETRY = True
        self.max_corr = config.max_points
        self.augment_shift_range = config.augment_shift_range
        self.augment_scale_max = config.augment_scale_max
        self.augment_scale_min = config.augment_scale_min

        # Initiate containers
        self.files = []
        self.kitti_icp_cache = {}
        self.kitti_cache = {}
        self.prepare_kitti_ply(split)
        self.split = split

    def prepare_kitti_ply(self, split):
        assert split in ['train', 'val', 'test']

        subset_names = open(self.DATA_FILES[split]).read().split()
        self.subset_names = subset_names
        for dirname in subset_names:
            drive_id = int(dirname)
            fnames = glob.glob(self.root + '/sequences/%02d/velodyne/*.bin' % drive_id)
            assert len(fnames) > 0, f"Make sure that the path {self.root} has data {dirname}"
            inames = sorted([int(os.path.split(fname)[-1][:-4]) for fname in fnames])

            # get one-to-one distance by comparing the translation vector
            all_odo = self.get_video_odometry(drive_id, return_all=True)
            all_pos = np.array([self.odometry_to_positions(odo) for odo in all_odo])
            Ts = all_pos[:, :3, 3]
            pdist = (Ts.reshape(1, -1, 3) - Ts.reshape(-1, 1, 3)) ** 2
            pdist = np.sqrt(pdist.sum(-1))

            ######################################
            # D3Feat script to generate test pairs
            more_than_10 = pdist > 10
            curr_time = inames[0]
            while curr_time in inames:
                next_time = np.where(more_than_10[curr_time][curr_time:curr_time + 100])[0]
                if len(next_time) == 0:
                    curr_time += 1
                else:
                    next_time = next_time[0] + curr_time - 1

                if next_time in inames:
                    self.files.append((drive_id, curr_time, next_time))
                    curr_time = next_time + 1

        # remove bad pairs
        if split == 'test':
            self.files.remove((8, 15, 58))
            self.augment_noise = False
        print(f'Num_{split}: {len(self.files)}')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        drive = self.files[idx][0]
        t0, t1 = self.files[idx][1], self.files[idx][2]
        all_odometry = self.get_video_odometry(drive, [t0, t1])
        positions = [self.odometry_to_positions(odometry) for odometry in all_odometry]
        fname0 = self._get_velodyne_fn(drive, t0)
        fname1 = self._get_velodyne_fn(drive, t1)

        # XYZ and reflectance
        xyzr0 = np.fromfile(fname0, dtype=np.float32).reshape(-1, 4)
        xyzr1 = np.fromfile(fname1, dtype=np.float32).reshape(-1, 4)

        xyz0 = xyzr0[:, :3]
        xyz1 = xyzr1[:, :3]

        # use ICP to refine the ground_truth pose, for ICP we don't voxllize the point clouds
        key = '%d_%d_%d' % (drive, t0, t1)
        filename = self.icp_path + '/' + key + '.npy'
        if key not in self.kitti_icp_cache:
            if not os.path.exists(filename):
                print('missing ICP files, recompute it')

                _, sel0 = ME.utils.sparse_quantize(xyz0 / 0.05, return_index=True)
                _, sel1 = ME.utils.sparse_quantize(xyz1 / 0.05, return_index=True)

                M = (self.velo2cam @ positions[0].T @ np.linalg.inv(positions[1].T)
                     @ np.linalg.inv(self.velo2cam)).T
                xyz0_t = apply_transform(xyz0[sel0], M)
                pcd0 = to_o3d_pcd(xyz0_t)
                pcd1 = to_o3d_pcd(xyz1[sel1])
                reg = open3d.registration.registration_icp(pcd0, pcd1, 0.2, np.eye(4),
                                                           open3d.registration.TransformationEstimationPointToPoint(),
                                                           open3d.registration.ICPConvergenceCriteria(
                                                               max_iteration=300))
                pcd0.transform(reg.transformation)
                M2 = M @ reg.transformation
                np.save(filename, M2)
            else:
                M2 = np.load(filename)
            self.kitti_icp_cache[key] = M2
        else:
            M2 = self.kitti_icp_cache[key]

        # refined pose is denoted as trans
        trans = M2
        #T_gt = np.linalg.inv(trans)
        R, T = decompose_rotation_translation(trans)

        euler = matrix_to_euler_angles(torch.from_numpy(R), "ZYX")

        # voxelize the point clouds here
        xyz0_th = torch.from_numpy(xyz0)
        xyz1_th = torch.from_numpy(xyz1)

        _, sel0 = ME.utils.sparse_quantize(xyz0_th / self.voxel_size, return_index=True)
        _, sel1 = ME.utils.sparse_quantize(xyz1_th / self.voxel_size, return_index=True)

        src_xyz = xyz0_th[sel0]
        tgt_xyz = xyz1_th[sel1]

        # Make point clouds using voxelized points
        pcd0 = make_open3d_point_cloud(xyz0[sel0])
        pcd1 = make_open3d_point_cloud(xyz1[sel1])

        # Get matches
        matching_inds = get_matching_indices(pcd0, pcd1, trans, self.matching_search_voxel_size)
        if len(matching_inds) < 1000:
            raise ValueError(f"{drive}, {t0}, {t1}")

        #src_over, tgt_over, over_index0, over_index1 = compute_overlap(src_xyz.numpy(), tgt_xyz.numpy(), self.matching_search_voxel_size)

        src_over, tgt_over, over_index0, over_index1 = ground_truth_attention(src_xyz, tgt_xyz, trans, self.matching_search_voxel_size)
        # Get features
        npts0 = len(sel0)
        npts1 = len(sel1)

        feats_train0, feats_train1 = [], []

        unique_xyz0_th = xyz0_th[sel0]
        unique_xyz1_th = xyz1_th[sel1]

        feats_train0.append(torch.ones((npts0, 1)))
        feats_train1.append(torch.ones((npts1, 1)))

        feats0 = torch.cat(feats_train0, 1)
        feats1 = torch.cat(feats_train1, 1)

        coords0 = torch.floor(unique_xyz0_th / self.voxel_size)
        coords1 = torch.floor(unique_xyz1_th / self.voxel_size)

        # add data augmentation
        src_pcd_input = copy.deepcopy(np.array(pcd0.points))
        tgt_pcd_input = copy.deepcopy(np.array(pcd1.points))

        if (self.data_augmentation):
            # add gaussian noise
            src_pcd_input += (np.random.rand(src_pcd_input.shape[0], 3) - 0.5) * self.augment_noise
            tgt_pcd_input += (np.random.rand(tgt_pcd_input.shape[0], 3) - 0.5) * self.augment_noise

            # rotate the point cloud
            euler_ab = np.random.rand(3) * np.pi * 2  # anglez, angley, anglex
            rot_ab = Rotation.from_euler('zyx', euler_ab).as_matrix()
            if (np.random.rand(1)[0] > 0.5):
                src_pcd_input = np.dot(rot_ab, src_pcd_input.T).T
            else:
                tgt_pcd_input = np.dot(rot_ab, tgt_pcd_input.T).T

            # scale the pcd
            scale = self.augment_scale_min + (self.augment_scale_max - self.augment_scale_min) * random.random()
            src_pcd_input = src_pcd_input * scale
            tgt_pcd_input = tgt_pcd_input * scale

            # shift the pcd
            shift_src = np.random.uniform(-self.augment_shift_range, self.augment_shift_range, 3)
            shift_tgt = np.random.uniform(-self.augment_shift_range, self.augment_shift_range, 3)

            src_pcd_input = src_pcd_input + shift_src
            tgt_pcd_input = tgt_pcd_input + shift_tgt

        # OverRegion
        over_matching_inds = overlap_get_matching_indices(to_o3d_pcd(src_over), to_o3d_pcd(tgt_over), trans, self.matching_search_voxel_size)
        # overlap
        src_over_coords, tgt_over_coords = np.floor(src_over / self.voxel_size), np.floor(tgt_over / self.voxel_size) 

        # get over feats
        src_over_feats = np.ones((src_over_coords.shape[0],1),dtype=np.float32)
        tgt_over_feats = np.ones((tgt_over_coords.shape[0],1),dtype=np.float32)

        #src_xyz, tgt_xyz = to_tensor(src_xyz).float(), to_tensor(tgt_xyz).float()
        src_over, tgt_over = to_tensor(src_over).float(), to_tensor(tgt_over).float()
        over_index0, over_index1 = to_tensor(over_index0).int(), to_tensor(over_index1).int()


        if self.split in ['train', 'val']:
            return src_pcd_input, tgt_pcd_input, coords0.int(), coords1.int(), feats0.float(), feats1.float(), src_over, tgt_over, src_over_coords, tgt_over_coords, src_over_feats, tgt_over_feats, over_index0, over_index1, matching_inds, over_matching_inds, trans, euler, scale

        elif self.split in ['test']:
            return drive, xyz0, xyz1, trans, filename

    @property
    def velo2cam(self):
        try:
            velo2cam = self._velo2cam
        except AttributeError:
            R = np.array([
                7.533745e-03, -9.999714e-01, -6.166020e-04, 1.480249e-02, 7.280733e-04,
                -9.998902e-01, 9.998621e-01, 7.523790e-03, 1.480755e-02
            ]).reshape(3, 3)
            T = np.array([-4.069766e-03, -7.631618e-02, -2.717806e-01]).reshape(3, 1)
            velo2cam = np.hstack([R, T])
            self._velo2cam = np.vstack((velo2cam, [0, 0, 0, 1])).T
        return self._velo2cam

    def get_video_odometry(self, drive, indices=None, ext='.txt', return_all=False):
        if self.IS_ODOMETRY:
            data_path = self.root + '/poses/%02d.txt' % drive
            if data_path not in self.kitti_cache:
                self.kitti_cache[data_path] = np.genfromtxt(data_path)
            if return_all:
                return self.kitti_cache[data_path]
            else:
                return self.kitti_cache[data_path][indices]

    def odometry_to_positions(self, odometry):
        if self.IS_ODOMETRY:
            T_w_cam0 = odometry.reshape(3, 4)
            T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
            return T_w_cam0

    def _get_velodyne_fn(self, drive, t):
        if self.IS_ODOMETRY:
            fname = self.root + '/sequences/%02d/velodyne/%06d.bin' % (drive, t)
        return fname

    def get_position_transform(self, pos0, pos1, invert=False):
        T0 = self.pos_transform(pos0)
        T1 = self.pos_transform(pos1)
        return (np.dot(T1, np.linalg.inv(T0)).T if not invert else np.dot(
            np.linalg.inv(T1), T0).T)