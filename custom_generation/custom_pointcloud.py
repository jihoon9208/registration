from math import degrees
import os 
import open3d as o3d
import numpy as np
import argparse
import time 
import torch

# Data argumentation requirement
from torch_points3d.core.data_transform import GridSampling3D, AddFeatByKey, AddOnes, Random3AxisRotation, GridSphereSampling, RandomSphere

from torch_geometric.transforms import Compose
from torch_points3d.datasets.registration.pair import Pair


def make_open3d_point_cloud(xyz, color=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if color is not None:
        if len(color) != len(xyz):
            color = np.tile(color, (len(xyz), 1))
        pcd.colors = o3d.utility.Vector3dVector(color)
    return pcd

def get_blue():
    """
    Get color blue for rendering
    """
    return [0, 0.651, 0.929]

def get_yellow():
    """
    Get color yellow for rendering
    """
    return [1, 0.706, 0]

def change_data(input):
    data = Pair(pos=torch.from_numpy(np.asarray(input.points)).float(), batch=torch.zeros(len(input.points)).long())
    return data

    
def main(config):
    root = config.input
    file_list = sorted(os.listdir(root))
    print ("file_list: {}".format(file_list))
    pcd_total = []
    trans_pcd = []
    count = 0
    for i in file_list:
        if "point" in i:
            pcd_numpy = []
            file = open(root + i, "r" )
            content = file.readlines()
            for j in content:
                test = j.split(',')
                test[2] = test[2].replace('\n', '')

                test_float = np.array(test).astype(float)
                pcd_numpy.append(test_float)
            pcd = make_open3d_point_cloud(pcd_numpy)
            #downpcd = pcd.voxel_down_sample(voxel_size=0.1)
            pcd_total.append(pcd)

    vis0 = o3d.visualization.Visualizer()
    vis0.create_window(window_name='Input0', width=960, height=540, left=0, top=0)

    vis1 = o3d.visualization.Visualizer()
    vis1.create_window(window_name='Input1', width=960, height=540, left=0, top=0)

    # Data argumentation
    # MS-SVConv, UDGE
    transfo_3dm = Compose(
        [
       
        Random3AxisRotation(rot_x=0, rot_y=0, rot_z=0),
        #RandomSphere(radius=0.2, strategy='random', class_weight_method='sqrt', center=True),
        GridSampling3D(size=0.3, quantize_coords=True, mode='mean'),
        #GridSphereSampling(radius=0.2, grid_size=0.2, delattr_kd_tree=True, center=True),
        AddOnes(), 
        AddFeatByKey(add_to_x=True, feat_name="ones")
        ]
    ) 

    for t in range(len(file_list)):
        trans_tmp = change_data(pcd_total[t])
        trans_pcd.append(transfo_3dm(trans_tmp))
    
    """ 
    data_source = change_data(pcd_total[0])
    data_target = change_data(pcd_total[1])
    data_s = transfo_3dm(data_source)
    data_t = transfo_3dm(data_target)

    while True:
        vis.add_geometry(pcd_total[0])
        vis.add_geometry(pcd_total[1])
        if not vis.poll_events():
            break
        vis.update_renderer()

        vis1.add_geometry(make_open3d_point_cloud(data_s.pos.numpy()))
        vis1.add_geometry(make_open3d_point_cloud(data_t.pos.numpy()))
        if not vis1.poll_events():
            break    
        vis1.update_renderer() """

    for j in range(len(file_list)):
        while True:
            tmp0 = pcd_total[j]    
            vis0.add_geometry(tmp0)

            vis0.update_geometry(tmp0)
            if not vis0.poll_events():
                break
            vis0.update_renderer()

            tmp1 = make_open3d_point_cloud(trans_pcd[j].pos.numpy())  
            vis1.add_geometry(tmp1)
      
            vis1.update_geometry(tmp1)
            if not vis1.poll_events():
                break
            vis1.update_renderer()
        
        vis1.capture_screen_image('./order_input/origin_image_{0:02d}.jpg'.format(count))
        time.sleep(1)
        vis0.capture_screen_image('./order_input/trans_image_{0:02d}.jpg'.format(count))
        count +=1
if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        '--input',
        default="../Datasets/DRONE1/total/point_cloud_00/",
        type=str,
        help='path to a pointcloud file')

    parser.add_argument(
        '-m',
        '--model',
        default='ResUNetBN2C-16feat-3conv.pth',
        type=str,
        help='path to latest checkpoint (default: None)')
    parser.add_argument(
        '--voxel_size',
        default=0.025,
        type=float,
        help='voxel size to preprocess point cloud')

    config = parser.parse_args()
    main(config)