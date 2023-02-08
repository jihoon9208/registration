
import os 
import open3d as o3d
import numpy as np
import argparse
import re



def make_open3d_point_cloud(xyz, color=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if color is not None:
        if len(color) != len(xyz):
            color = np.tile(color, (len(xyz), 1))
        pcd.colors = o3d.utility.Vector3dVector(color)
    return pcd

def main(config):
    root = config.input
    file_list = sorted(os.listdir(root))
    print ("file_list: {}".format(file_list))

    for i in file_list:
        if ".txt" in i:
            file = open(root + i , "r") 
            content = file.readlines()
            
            count = 1
            file_count = 0
            file_list = []
            point_list = []

            for j in content:
                test = j.split(',')
                test[2] = test[2].replace('\n', '')
                file_list.append(np.array(test, dtype=np.float32))

            pcd = make_open3d_point_cloud(file_list)
            #downpcd = pcd.voxel_down_sample(voxel_size=0.1)
            #o3d.visualization.draw_geometries([downpcd])
            length = len(pcd.points)

            pcd_tree = o3d.geometry.KDTreeFlann(pcd)
            
            """ for z in range(length):
                if z % 100000 == 0:

                    [_, idx0, _] = pcd_tree.search_radius_vector_3d(pcd.points[z], 10)

                    if not os.path.exists(root + i.split('.')[0]):
                            os.mkdir(root + i.split('.')[0] )
                    file0 = open(root + i.split('.')[0] + "/" + 'point_cloud_' + "{0:2d}".format(file_count) +'.txt' , "w") 
                    
                    for t in idx0:
                        point_list.append(np.array(pcd.points[t], dtype=np.float32))

                    for d in point_list:
                        file0.write(str(d[0])+','+str(d[1])+','+str(d[2]))
                        file0.write('\n')
                    file0.close()
                    file_count += 1
                    z += 20000
                point_list = [] """

            for z in range(length):
                point_list.append(pcd.points[z])

                if count % 100000 == 0:
                    #point_list = np.sort(point_list)
                    if not os.path.exists(root + i.split('.')[0]):
                        os.mkdir(root + i.split('.')[0] )
                    file0 = open(root + i.split('.')[0] + "/" + 'point_cloud_' + "{0:02d}".format(file_count) +'.txt' , "w") 
                    
                    for d in point_list:
                        file0.write(str(d[0])+','+str(d[1])+','+str(d[2]))
                        file0.write('\n')

                    count = 1
                    point_list=[]
                    z -= 70000
                    file0.close()
                    file_count += 1
                
                count += 1
                
     

                    
if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        '--input',
        default="../Datasets/DRONE/total/",
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