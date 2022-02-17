# Part of the code in this file is taken from https://github.com/chrischoy/DeepGlobalRegistration/blob/46dd264580b4191accedc277f4ae434acdb4d380/dataloader/base_loader.py#L24


import torch
import numpy as np
import MinkowskiEngine as ME

from tools.utils import to_tensor 

class CollateFunc:

    def __init__(self):
        self.collation_fn = self.collate_pair_fn

    def __call__(self, list_data):
        return self.collation_fn(list_data)

    def collate_pair_fn(list_data):
        xyz0, xyz1, coords0, coords1, feats0, feats1, xyz0_over, xyz1_over, over_index0, over_index1, matching_inds, rot, trans, euler, scale = list(
            zip(*list_data))

        # prepare inputs for FCGF
        src_batch_C, src_batch_F = ME.utils.sparse_collate(coords0, feats0)
        tgt_batch_C, tgt_batch_F = ME.utils.sparse_collate(coords1, feats1)

        # concatenate xyz
        src_xyz = torch.cat(xyz0, 0).float()
        tgt_xyz = torch.cat(xyz1, 0).float()

        src_over_xyz = torch.cat(xyz0_over, 0).float()
        tgt_over_xyz = torch.cat(xyz1_over, 0).float()

        over_index0 = torch.cat(over_index0, 0).int()
        over_index1 = torch.cat(over_index1, 0).int()

        # add batch indice to matching_inds
        matching_inds_batch = []
        len_batch = []
        curr_start_ind = torch.zeros((1,2))

        for batch_id, _ in enumerate(matching_inds):
            N0 = coords0[batch_id].shape[0]
            N1 = coords1[batch_id].shape[0]
            matching_inds_batch.append(matching_inds[batch_id]+curr_start_ind)
            len_batch.append([N0,N1])

            curr_start_ind[0,0]+=N0
            curr_start_ind[0,1]+=N1   

        matching_inds_batch = torch.cat(matching_inds_batch, 0).int()

        return {
            'pcd0': src_xyz,
            'pcd1': tgt_xyz,
            'pcd0_over' : src_over_xyz,
            'pcd1_over' : tgt_over_xyz,
            'over_index0' : over_index0,
            'over_index1' : over_index1,
            'sinput0_C': src_batch_C,
            'sinput0_F': src_batch_F.float(),
            'sinput1_C': tgt_batch_C,
            'sinput1_F': tgt_batch_F.float(),
            'correspondences': matching_inds_batch,
            'rot': rot[0],
            'trans': trans[0],
            'scale': scale[0],
            'Euler_gt': euler[0],
            'len_batch': len_batch
        }
