import torch
import MinkowskiEngine as ME
import numpy as np

from tools.pointcloud import ground_truth_attention

def datasets_setting(pcd0, pcd1, T_gt, voxel_size, search_voxel_size, device):

    feats0 = []
    feats1 = []
    
    feats0.append(np.ones((len(pcd0), 1)))
    feats1.append(np.ones((len(pcd1), 1)))

    feats0 = np.hstack(feats0)
    feats1 = np.hstack(feats1)

    # Voxelize xyz and feats
    coords0 = np.floor(pcd0 / voxel_size)
    coords0, inds0 = ME.utils.sparse_quantize(coords0, return_index=True)
    # Convert to batched coords compatible with ME
    coords0 = ME.utils.batched_coordinates([coords0])
    return_coords = pcd0[inds0]

    feats0 = feats0[inds0]

    feats0 = torch.tensor(feats0, dtype=torch.float32)
    coords0 = coords0.clone().detach()

    sinput0 = ME.SparseTensor(feats0, coordinates=coords0, device=device)

    # Voxelize xyz and feats
    coords1 = np.floor(pcd1 / voxel_size)
    coords1, inds1 = ME.utils.sparse_quantize(coords1, return_index=True)
    # Convert to batched coords compatible with ME
    coords1 = ME.utils.batched_coordinates([coords1])
    return_coords = pcd1[inds1]

    feats1 = feats1[inds1]

    feats1 = torch.tensor(feats1, dtype=torch.float32)
    coords1 = coords1.clone().detach()

    sinput1 = ME.SparseTensor(feats1, coordinates=coords1, device=device)

    src_over, tgt_over, over_index0, over_index1 = ground_truth_attention(pcd0[inds0], pcd1[inds1], T_gt, search_voxel_size)

    return sinput0, sinput1, src_over, tgt_over, over_index0, over_index1