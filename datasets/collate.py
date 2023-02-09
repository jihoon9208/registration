# Part of the code in this file is taken from https://github.com/chrischoy/DeepGlobalRegistration/blob/46dd264580b4191accedc277f4ae434acdb4d380/dataloader/base_loader.py#L24


import torch
import numpy as np

from tools.utils import to_tensor 

class CollateFunc:

    def __init__(self):
        self.collation_fn = self.collate_pair_fn

    def __call__(self, list_data):
        return self.collation_fn(list_data)

    def collate_pair_fn(list_data):
        xyz0, xyz1, coords0, coords1, feats0, feats1, xyz0_over, xyz1_over, \
        over_index0, over_index1, matching_inds, over_matching_inds, \
        trans, euler, scale = list(zip(*list_data))

        src_batch0, tgt_batch1 = [], []
        src_over_xyz, tgt_over_xyz = [], [] 
        src_over_index, tgt_over_index = [], []

        # add batch indice to matching_inds
        matching_inds_batch, over_matching_inds_batch = [], []
        match_inds_batch, match_num_batch = [], []
        len_batch = []

        curr_start_inds = np.zeros((1, 2))

        for batch_id, _ in enumerate(matching_inds):
            N0 = coords0[batch_id].shape[0]
            N1 = coords1[batch_id].shape[0]
            
            #######################
            # full
            src_batch0.append(to_tensor(xyz0[batch_id]))
            tgt_batch1.append(to_tensor(xyz1[batch_id]))

            #######################
            # overlap
            src_over_xyz.append(to_tensor(xyz0_over[batch_id]))
            tgt_over_xyz.append(to_tensor(xyz1_over[batch_id]))

            src_over_index.append(over_index0[batch_id].long())
            tgt_over_index.append(over_index1[batch_id].long())

            # correspondence 
            matching_inds_batch.append(torch.from_numpy(np.array(matching_inds[batch_id]) + curr_start_inds))
            over_matching_inds_batch.append(torch.from_numpy(np.array(over_matching_inds[batch_id])))

            # correspondence of each batch 
            match_inds_batch.append(torch.from_numpy(np.array(matching_inds[batch_id])))
            match_num_batch.append(len(matching_inds[batch_id]))

            len_batch.append([N0,N1])

            curr_start_inds[0,0]+=N0
            curr_start_inds[0,1]+=N1   

        return {
            'pcd0': src_batch0,
            'pcd1': tgt_batch1,
            'pcd0_over' : src_over_xyz,
            'pcd1_over' : tgt_over_xyz,
            'over_index0' : over_index0,
            'over_index1' : over_index1,
            'sinput0_C': coords0,
            'sinput0_F': feats0,
            'sinput1_C': coords1,
            'sinput1_F': feats1,    
            'correspondences': matching_inds_batch,
            'T_gt': trans,
            'scale': scale,
            'Euler_gt': euler,
            'len_batch': len_batch,
            'pos_pair': match_inds_batch,
            'pos_len': match_num_batch,
        }