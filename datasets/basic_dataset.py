
from torch.utils.data import Dataset
import numpy as np
import logging

class BasicDataset(Dataset):
    AUGMENT = None

    def __init__(self,
                phase,
                transform=None,
                random_rotation=True,
                random_scale=False,
                manual_seed=False,
                config=None):
        self.phase = phase
        self.files = []
        self.data_objects = []
        
        self.voxel_size = config.voxel_size
        self.random_scale = random_scale
        self.min_scale = config.min_scale
        self.max_scale = config.max_scale

        self.random_rotation = random_rotation
        self.rotation_range = config.rotation_range
        self.randg = np.random.RandomState()
        
        if manual_seed:
            self.reset_seed()

    def reset_seed(self, seed=0):
        logging.info(f"Resetting the data loader seed to {seed}")
        self.randg.seed(seed)

    def __len__(self):
        return len(self.files)
