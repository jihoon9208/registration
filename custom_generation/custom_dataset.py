from math import degrees
from operator import length_hint
import os 
import open3d as o3d
import numpy as np
import argparse
from easydict import EasyDict as edict
from subprocess import run

def main(config):
    root = config.input
    file_list = sorted(os.listdir(root))
    print ("file_list: {}".format(file_list))

    for i in file_list:
        if ".txt" in i:
            file = open(root + i , "r") 
            content = file.readlines()
            split_content = content[0].split('},')

            count = 1
            file_count = 0
            file_list = []
            length = len(split_content)
            tmp = 0

            for idx, j in enumerate(split_content):
                test = j.split(',')
                file_list.append(test[1].split(":")[1].replace(" ",'')+ ',' + test[2].split(":")[1] + ',' + test[3].split(":")[1])

                if count % 4000000 == 0:
                    file0 = open(root + 'point_cloud_' + "{0:02d}".format(file_count) +'.txt' , "w") 
                    for t in file_list:
                        file0.write(str(t))
                        file0.write('\n')
                    file_count += 1
                    file0.close()
                    file_list = []
                    count = 1

                elif idx == length-1 :
                    file0 = open(root + 'point_cloud_' + "{0:02d}".format(file_count) +'.txt' , "w") 
                    for t in file_list:
                        file0.write(str(t))
                        file0.write('\n')
                    file_count += 1
                    file0.close()

                count += 1
                continue
                    
if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        '--input',
        default="../Datasets/DRONE1/",
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