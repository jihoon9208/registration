from math import degrees
import os 
import open3d as o3d
import numpy as np
import argparse
import time 


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
    
def main(config):
    root = config.input
    file_list = sorted(os.listdir(root))
    sort_file_list = []
    for i in file_list :
        if ".txt" in i :
            sort_file_list.append(i)
                    
    print ("file_list: {}".format(sort_file_list))
    pcd_total = []
    pcd_part = []
    
    tmp_list = []
    count = 0
    file_coout = 1
    for i in sort_file_list:
        if ".txt" in i:

            pcd_numpy = []
            file = open(root + i, "r" )
            content = file.readlines()
            for j in content:
                test = j.split(',')
                if ']' in test[2]:
                    test[2] = test[2].replace('}]\n', '')
                else :
                    test[2] = test[2].replace('\n', '')

                test_float = np.array(test).astype(float)
                pcd_numpy.append(test_float)

            pcd = make_open3d_point_cloud(pcd_numpy)
            #downpcd = pcd.voxel_down_sample(voxel_size=0.1)
            pcd_total.append(pcd)

            """ for j in range(len(pcd.points)):
                pcd_part.append(pcd.points[j])
                if file_coout % 100000 == 0:
                    tmp_pcd = make_open3d_point_cloud(pcd_part)
                    tmp_list.append(tmp_pcd)

                    tmp_pcd = 0
                file_coout += 1
                

            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name='Input', width=1920, height=1080, left=0, top=0)
            for j in range(len(tmp_list)):
                
                while True:
                    tmp = tmp_list[j]
                    vis.add_geometry(tmp)
                    time.sleep(1)
                    vis.update_geometry(tmp)
                    if not vis.poll_events():
                        break
                    vis.update_renderer()
                vis.capture_screen_image('./part_image/image_{0:02d}.jpg'.format(count))
                file_coout = 1
                count +=1
                pcd_part = []
                
            vis.destroy_window() """


    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Input', width=1920, height=1080, left=0, top=0)
    
    for j in range(len(sort_file_list)):
        while True:
            tmp = pcd_total[j]    
            vis.add_geometry(tmp)
            time.sleep(1)
            vis.update_geometry(tmp)
            if not vis.poll_events():
                break
            vis.update_renderer()
        vis.capture_screen_image('./total_image/image_{0:02d}.jpg'.format(count))
        count +=1
if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        '--input',
        default="../Datasets/DRONE1/total/",
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