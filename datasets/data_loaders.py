# -*- coding: future_fstrings -*-

import logging

import torch
import torch.utils.data

import tools.transforms as t
from.collate import CollateFunc as coll
from tools.utils import load_obj
from .threedmatch_dataset import ThreeDMatchPairDataset, ThreeDMatchTestDataset, ThreeDLoMatchTestDataset
from .kitti_dataset import KITTINMPairDataset, KITTIPairDataset


ALL_DATASETS = [ThreeDMatchPairDataset, ThreeDMatchTestDataset, ThreeDLoMatchTestDataset, KITTIPairDataset, KITTINMPairDataset]
dataset_str_mapping = {d.__name__: d for d in ALL_DATASETS}


def make_data_loader(config, phase, batch_size, num_threads=0, shuffle=None):
    assert phase in ['train', 'trainval', 'val', 'test']
    if shuffle is None:
        shuffle = phase != 'test'


    if config.dataset not in dataset_str_mapping.keys():
        logging.error(f'Dataset {config.dataset}, does not exists in ' +
                    ', '.join(dataset_str_mapping.keys()))
        
    Dataset = dataset_str_mapping[config.dataset]

    use_random_scale = False
    use_random_rotation = False
    transforms = []
    if phase in ['train', 'val']:
        use_random_rotation = config.use_random_rotation
        use_random_scale = config.use_random_scale
        transforms += [t.Jitter()]        

    if phase in ['train', 'val']:
        dset = Dataset(
            phase,
            transform=t.Compose(transforms),
            random_scale=use_random_scale,
            random_rotation=use_random_rotation,
            manual_seed=False,
            config=config)
    
    else : 
        dset = Dataset(
            config.source)

    if phase in ['train', 'val']:

        loader = torch.utils.data.DataLoader(
            dset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_threads,
            collate_fn=coll.collate_pair_fn,
            pin_memory=False,
            drop_last=True)

    else :
        loader = torch.utils.data.DataLoader(
            dset, batch_size=1, num_workers=1, shuffle=False, collate_fn=lambda x: x
    )
    return loader


# Predator github modified
def get_datasets(config):

    if(config.dataset=='ThreeDMatchPairDataset'):
        info_train = load_obj(config.train_info)
        info_val = load_obj(config.val_info)
        info_benchmark = load_obj(config.test_full_info)

        train_set = ThreeDMatchPairDataset(info_train,config)
        val_set = ThreeDMatchPairDataset(info_val,config)
        benchmark_set = ThreeDMatchPairDataset(info_benchmark,config)

    elif(config.dataset == 'kitti'):
        train_set = KITTINMPairDataset(config,'train',data_augmentation=True)
        val_set = KITTINMPairDataset(config,'val',data_augmentation=False)
        benchmark_set = KITTINMPairDataset(config, 'test',data_augmentation=False)
    else:
        raise NotImplementedError

    return train_set, val_set, benchmark_set