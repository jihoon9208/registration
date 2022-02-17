import torch
from torch.utils.data import Dataset
import numpy as np
import logging
from sklearn.neighbors import NearestNeighbors

class BasicDataset(Dataset):
    AUGMENT = None

    def __init__(self,
                phase,
                transform=None,
                random_rotation=True,
                random_scale=True,
                manual_seed=False,
                config=None):
        self.phase = phase
        self.files = []
        self.data_objects = []
        self.transform = transform
        self.voxel_size = config.voxel_size
        self.matching_search_voxel_size = \
            config.voxel_size * config.positive_pair_search_voxel_size_multiplier

        self.random_scale = random_scale
        self.min_scale = config.min_scale
        self.max_scale = config.max_scale
        self.num_points = config.num_points

        self.random_rotation = random_rotation
        self.rotation_range = config.rotation_range
        self.randg = np.random.RandomState()
        if manual_seed:
            self.reset_seed()

    def reset_seed(self, seed=0):
        logging.info(f"Resetting the data loader seed to {seed}")
        self.randg.seed(seed)

    def sample_points(self, pts, num_points):
        if pts.shape[0] > num_points:
            pts = np.random.permutation(pts)[:num_points]
        else:
            pts = np.random.permutation(pts)
        return pts

    def apply_transform(self, pts, trans):
        R = trans[:3, :3]
        T = trans[:3, 3]
        pts = pts @ R.T + T
        return pts

    def ground_truth_attention(self, p1, p2, trans):
        
        p1 = self.sample_points(p1, self.num_points)
        p2 = self.sample_points(p2, self.num_points)

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

    def __len__(self):
        return len(self.files)
