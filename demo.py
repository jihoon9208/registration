"""
Scripts for pairwise registration demo

Author: Shengyu Huang
Last modified: 22.02.2021
"""
import os, torch, time, shutil, json,glob,sys,copy, argparse
import numpy as np
from easydict import EasyDict as edict
from torch.utils.data import Dataset
from torch import optim, nn
import open3d as o3d

from rich.console import Console
from rich.progress import track
from rich.table import Table

from config_test import get_config
from model.network import PoseEstimator
from model.simpleunet import SimpleNet

from datasets.data_loaders import make_data_loader

from tools.file import ensure_dir
from datasets.threedmatch_dataset import ThreeDMatchTestDataset
from tools.test_utils import datasets_setting
from tools.pointcloud import ground_truth_attention
from lib.benchmark_utils import ransac_pose_estimation, to_o3d_pcd, get_blue, get_yellow, to_tensor, get_white

import shutil

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

class ThreeDMatchDemo(Dataset):
    """
    Load subsampled coordinates, relative rotation and translation
    Output(torch.Tensor):
        src_pcd:        [N,3]
        tgt_pcd:        [M,3]
        rot:            [3,3]
        trans:          [3,1]
    """
    def __init__(self,config, src_path ):
        super(ThreeDMatchDemo,self).__init__()
        self.config = config
        self.src_path = src_path
        #self.tgt_path = tgt_path

    def __len__(self):
        return 1

    def __getitem__(self,item): 
        # get pointcloud
        src_pcd = torch.load(self.src_path).astype(np.float32)
        tgt_pcd = torch.load(self.tgt_path).astype(np.float32)
        
        #src_pcd = o3d.io.read_point_cloud(self.src_path)
        #tgt_pcd = o3d.io.read_point_cloud(self.tgt_path)
        #src_pcd = src_pcd.voxel_down_sample(0.025)
        #tgt_pcd = tgt_pcd.voxel_down_sample(0.025)
        #src_pcd = np.array(src_pcd.points).astype(np.float32)
        #tgt_pcd = np.array(tgt_pcd.points).astype(np.float32)

        src_feats=np.ones_like(src_pcd[:,:1]).astype(np.float32)
        tgt_feats=np.ones_like(tgt_pcd[:,:1]).astype(np.float32)

        # fake the ground truth information
        rot = np.eye(3).astype(np.float32)
        trans = np.ones((3,1)).astype(np.float32)
        correspondences = torch.ones(1,2).long()

        return src_pcd,tgt_pcd,src_feats,tgt_feats,rot,trans, correspondences, src_pcd, tgt_pcd, torch.ones(1)


def lighter(color, percent):
    '''assumes color is rgb between (0, 0, 0) and (1,1,1)'''
    color = np.array(color)
    white = np.array([1, 1, 1])
    vector = white-color
    return color + vector * percent


def draw_registration_result(src_raw, tgt_raw, src_index, tgt_index, tsfm):
    ########################################
    # 1. input point cloud
    src_pcd_before = to_o3d_pcd(src_raw)
    tgt_pcd_before = to_o3d_pcd(tgt_raw)
    src_pcd_before.paint_uniform_color(get_yellow())
    tgt_pcd_before.paint_uniform_color(get_blue())
    src_pcd_before.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))
    tgt_pcd_before.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))

    ########################################
    # 2. overlap colors
    rot, trans = to_tensor(tsfm[:3,:3]), to_tensor(tsfm[:3,3][:,None])
    src_overlap = to_o3d_pcd(src_raw[src_index])
    tgt_overlap = to_o3d_pcd(tgt_raw[tgt_index])
    src_overlap.paint_uniform_color(get_yellow())
    tgt_overlap.paint_uniform_color(get_blue())
    src_overlap.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))
    tgt_overlap.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))

    src_overlap.transform(tsfm)
    #src_overlap_color = lighter(get_yellow(), src_overlap)
    #tgt_overlap_color = lighter(get_blue(), tgt_overlap)

    src_pcd_overlap = copy.deepcopy(src_pcd_before)
    src_pcd_overlap.transform(tsfm)
    tgt_pcd_overlap = copy.deepcopy(tgt_pcd_before)    
    src_pcd_overlap.paint_uniform_color(get_white())
    tgt_pcd_overlap.paint_uniform_color(get_white())
    """ src_pcd_overlap.colors = o3d.utility.Vector3dVector(src_overlap_color)
    tgt_pcd_overlap.colors = o3d.utility.Vector3dVector(tgt_overlap_color) """ 


    ########################################
    # 3. draw registrations
    src_pcd_after = copy.deepcopy(src_pcd_before)
    src_pcd_after.transform(tsfm)

    vis1 = o3d.visualization.Visualizer()
    vis1.create_window(window_name='Input', width=960, height=540, left=0, top=0)
    vis1.add_geometry(src_pcd_before)
    vis1.add_geometry(tgt_pcd_before)

    vis2 = o3d.visualization.Visualizer()
    vis2.create_window(window_name='Inferred overlap region', width=960, height=540, left=0, top=600)
    vis2.add_geometry(src_pcd_overlap)
    vis2.add_geometry(tgt_pcd_overlap)
    vis2.add_geometry(src_overlap)
    vis2.add_geometry(tgt_overlap)

    vis3 = o3d.visualization.Visualizer()
    vis3.create_window(window_name ='Our registration', width=960, height=540, left=960, top=0)
    vis3.add_geometry(src_pcd_after)
    vis3.add_geometry(tgt_pcd_before)
    
    while True:
        vis1.update_geometry(src_pcd_before)
        vis3.update_geometry(tgt_pcd_before)
        if not vis1.poll_events():
            break
        vis1.update_renderer()

        vis2.update_geometry(src_pcd_overlap)
        vis2.update_geometry(tgt_pcd_overlap)
        vis2.update_geometry(src_overlap)
        vis2.update_geometry(tgt_overlap)
        if not vis2.poll_events():
            break
        vis2.update_renderer()
        vis3.update_geometry(src_pcd_after)
        vis3.update_geometry(tgt_pcd_before)
        
        if not vis3.poll_events():
            break
        vis3.update_renderer()

        
    vis1.destroy_window()
    vis2.destroy_window()
    vis3.destroy_window()   

def rte_rre(T_pred, T_gt, rte_thresh, rre_thresh, eps=1e-8):
    if T_pred is None:
        return np.array([0, np.inf, np.inf])

    rte = np.linalg.norm(T_pred[:3, 3] - T_gt[:3, 3]) * 100
    rre = (
        np.arccos(
            np.clip(
                (np.trace(T_pred[:3, :3].T @ T_gt[:3, :3]) - 1) / 2, -1 + eps, 1 - eps
            )
        )
        * 180
        / np.pi
    )
    return np.array([rte < rte_thresh and rre < rre_thresh, rte, rre])


def main(config, demo_loader, model):

    model.eval()
    voxel_size=config.voxel_size

    overlap = config.voxel_size * config.positive_pair_search_voxel_size_multiplier

    tot_num_data = len(demo_loader)
    c_loader_iter = demo_loader.__iter__()

    pose_estimate = PoseEstimator(config).to(device)

    TE_THRESH = 30
    RE_THRESH = 15

    with torch.no_grad():
        for batch_idx in track(range(tot_num_data)):
            batch = c_loader_iter.next()
            sname, xyz0, xyz1, trans, f0, f1 = batch[0]
            T_gt = np.linalg.inv(trans)

            sinput0, sinput1, src_over, tgt_over, over_index0, over_index1 = datasets_setting(xyz0, xyz1, T_gt, voxel_size, overlap, device)

            start = time.time()            
            T, _, _ = pose_estimate(sinput0, sinput1, torch.from_numpy(src_over).to(device), torch.from_numpy(tgt_over).to(device), over_index0, over_index1, model.to(device))
            end = time.time()
            
            result = rte_rre(T, T_gt, TE_THRESH, RE_THRESH)

            recall = str(round(result[0],2))
            rte = str(round(result[1],2))
            rre = str(round(result[2],2))

            filename0 = f0.split('/')[-1]
            filename1 = f1.split('/')[-1]

            """ if float(recall) == 1 and float(rte) < 2 and float(rre) < 1.5 :
                with open("./data_list.txt", "a") as f:
                    f.write("filename0 : " + sname + "/" + filename0 + '\n' 
                        "filename1 : " + sname + "/" + filename1 + '\n'
                        + "recall : " + recall + " RTE : " + rte +  " RRE :" + rre + '\n')
           
            """                
            if sname=="7-scenes-redkitchen" and filename0=="cloud_bin_23.ply" and filename1=="cloud_bin_39.ply":
            
                draw_registration_result(xyz0, xyz1, over_index0, over_index1, T)


if __name__ == '__main__':
    # load configs
    
    config = get_config()
    config = vars(config)
    config = edict(config)

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = SimpleNet(
            conv1_kernel_size=config.conv1_kernel_size,
            D=6)

    checkpoint = torch.load(config.model)

    mconfig = checkpoint['config']
    model.load_state_dict(checkpoint['state_dict'])

    # create dataset and dataloader
    test_loader = make_data_loader(
        config,
        config.test_phase,
        config.val_batch_size,
        num_threads=config.val_num_thread)

    # do pose estimation
    main(config, test_loader, model)