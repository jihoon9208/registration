import logging
import os
from abc import ABC


import numpy as np
import open3d as o3d
import torch

from model import load_model


class BaseFeatureExtractor(ABC):
    def __init__(self):
        logging.info(f"Initialize {self.__class__.__name__}")

    def extract_feature(self, xyz):
        raise NotImplementedError("Feature should implement extract_feature method.")

class FCGF(BaseFeatureExtractor):
    def __init__(self, config ):
        super().__init__()
        self.voxel_size = config.voxel_size
        checkpoint_path = config.feat_weight
        assert os.path.exists(checkpoint_path), f"{checkpoint_path} not exists"

        MODEL = load_model("ResUNetBN2C")
        feat_model = MODEL(
            1, 32, bn_momentum=0.09, conv1_kernel_size=7, normalize_feature=True, 
        ).cuda()
        checkpoint = torch.load(checkpoint_path)
        feat_model.load_state_dict(checkpoint["state_dict"])

        self.feat_model = feat_model
        self.feat_model.eval()

    def freeze(self):
        for param in self.feat_model.parameters():
            param.requires_grad = False
    @ torch.no_grad()
    def extract_feature(self, sinput):

        # extract feature.
        F = self.feat_model(sinput).F

        return F

class FPFH(BaseFeatureExtractor):
    def __init__(self, voxel_size):
        super().__init__()

        self.voxel_size = voxel_size

    def extract_feature(self, xyz):
        voxel_size = self.voxel_size
        if isinstance(xyz, torch.Tensor):
            xyz = xyz.numpy()

        # downsample
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        #pcd = pcd.voxel_down_sample(voxel_size)

        # calculate normals
        radius_normal = voxel_size * 2.0
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
        )

        # calculate features
        radius_feature = voxel_size * 5.0
        # pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature
        pcd_fpfh = o3d.registration.compute_fpfh_feature(
            pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
        )
        xyz = torch.from_numpy(np.asarray(pcd.points)).float()
        F = torch.from_numpy(pcd_fpfh.data.copy().T).float().contiguous()
        return F, xyz


class Predator(BaseFeatureExtractor):
    def __init__(self, config ):

        super().__init__()
        self.voxel_size = config.voxel_size
        checkpoint_path = config.feat_weight
        assert os.path.exists(checkpoint_path), f"{checkpoint_path} not exists"

        MODEL = load_model("ResUNetBN2C")
        feat_model = MODEL(
            config, D=3
        ).cuda()

        checkpoint = torch.load(checkpoint_path)
        feat_model.load_state_dict(checkpoint["state_dict"])

        self.feat_model = feat_model
        self.feat_model.eval()

    @ torch.no_grad()
    def extract_feature(self, sinput0, sinput1):
        
        F0, F1 = self.feat_model(sinput0, sinput1)
        return F0, F1

MODELS = [FPFH, FCGF, Predator]



def get_feature(name):
    # Find the model class from its name
    all_models = MODELS
    mdict = {model.__name__: model for model in all_models}
    if name not in mdict:
        logging.info(f"Invalid model index. You put {name}. Options are:")
        # Display a list of valid model names
        for model in all_models:
            logging.info("\t* {}".format(model.__name__))
        return None
    NetClass = mdict[name]

    return NetClass
