import os
import os.path as osp
import gc
import logging
import numpy as np
import json
from tqdm import tqdm
import open3d as o3d

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd.grad_mode import no_grad
from tensorboardX import SummaryWriter
import MinkowskiEngine as ME
from sklearn.neighbors import NearestNeighbors
from tools.pointcloud import draw_registration_result, make_open3d_point_cloud

#from model import load_model
import model
from tools.file import ensure_dir
from tools.utils import validate_gradient, to_array, to_tsfm, Logger, to_tensor
from tools.model_util import npmat2euler, rotationMatrixToEulerAngles,transform_point_cloud, transform_point_cloud0 ,cdist_torch
from tools.transforms import apply_transform, decompose_rotation_translation

from lib.timer import Timer, AverageMeter
from lib.eval import find_nn_gpu
from tools.transforms import sample_points



class TrainerInit:

    def __init__(
        self,
        config,
        data_loader,
        val_data_loader=None,
        test_data_loader=None,
        model = None,
        optimizer = None,
        scheduler = None,
        loss = None     
    ):
        num_feats = 1  # occupancy only for 3D Match dataset. For ScanNet, use RGB 3 channels.

        # Model initialization

        if config.weights:
            checkpoint = torch.load(config.weights)
            model.load_state_dict(checkpoint['state_dict_F'])
        
        logging.info(model)

        if config.use_gpu and not torch.cuda.is_available():
            logging.warning('Warning: There\'s no CUDA support on this machine, '
                        'training is performed on CPU.')
            raise ValueError('GPU not available, but cuda flag set')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.get_loss = loss
        self.voxel_size = config.voxel_size
        self.max_epoch = config.max_epoch
        self.save_freq = config.save_freq_epoch
        self.batch_size = config.batch_size
        self.verbose = config.verbose

        self.val_max_iter = config.val_max_iter
        self.val_epoch_freq = config.val_epoch_freq
        self.verbose_freq= config.verbose_freq

        self.best_loss = 1e5
        self.best_recall = -1e5

        self.best_val_metric = config.best_val_metric
        self.best_val_epoch = -np.inf
        self.best_val = -np.inf
        self.start_epoch = 1
        self.checkpoint_dir = config.out_dir

        ensure_dir(self.checkpoint_dir)
        json.dump(
            config,
            open(os.path.join(self.checkpoint_dir, 'config.json'), 'w'),
            indent=4,
            sort_keys=False)

        self.iter_size = config.iter_size
        self.batch_size = data_loader.batch_size
        self.data_loader = data_loader
        self.val_data_loader = val_data_loader

        self.test_valid = True if self.val_data_loader is not None else False
        self.log_step = int(np.sqrt(self.config.batch_size))
        self.model = self.model.to(self.device)
        self.writer = SummaryWriter(logdir=config.out_dir)
        self.logger = Logger(config.snapshot_dir)
        self.logger.write(f'#parameters {sum([x.nelement() for x in self.model.parameters()])/1000000.} M\n')

        self.get_loss = loss

        if config.resume is not None:
            if osp.isfile(config.resume):
                logging.info("=> loading checkpoint '{}'".format(config.resume))
                state = torch.load(config.resume)
                self.start_epoch = state['epoch']
                model.load_state_dict(state['state_dict'])
                self.scheduler.load_state_dict(state['scheduler'])
                self.optimizer.load_state_dict(state['optimizer'])
                self.best_loss = state['best_val_loss']
                self.best_val = state['best_val']
                self.best_val_metric = state['best_val_metric']
                self.best_val_epoch = state['best_val_epoch']

            else:
                raise ValueError(f"=> no checkpoint found at '{config.resume}'")

    def train(self):
        """
        Full training logic
        """
        # Baseline random feature performance
        if self.test_valid:
            with torch.no_grad():
                val_dict = self.inference_one_epoch(0, 'val')

            for k, v in val_dict.items():
                self.writer.add_scalar(f'val/{k}', v.avg, 0 )
                print(k, v.avg)
        
        for epoch in range(self.start_epoch, self.max_epoch + 1):

            lr = self.scheduler.get_lr()
            logging.info(f" Epoch: {epoch}, LR: {lr}")
            #with torch.autograd.set_detect_anomaly(True):
            self.inference_one_epoch(epoch, 'train')
            self._save_checkpoint(epoch)
            self.scheduler.step()
            
            if self.test_valid and epoch % self.val_epoch_freq == 0:
                with torch.no_grad():
                    val_dict = self.inference_one_epoch(epoch, 'val')

                for k, v in val_dict.items():
                    print(k, v.avg)
                
                if self.best_loss > val_dict['total_loss'].avg :
                    self.best_loss = val_dict['total_loss'].avg
                    self._save_checkpoint(epoch, 'best_loss')

                if self.best_val < val_dict[self.best_val_metric].avg :
                    logging.info(
                        f'Saving the best val model with {self.best_val_metric}: {val_dict[self.best_val_metric].avg}'
                    )
                    self.best_val = val_dict[self.best_val_metric].avg
                    self.best_val_epoch = epoch
                    self._save_checkpoint(epoch, 'best_val_checkpoint')
                else:
                    logging.info(
                        f'Current best val model with {self.best_val_metric}: {self.best_val} at epoch {self.best_val_epoch}'
                    )

    def eval(self):
        print('Start to evaluate on validation datasets...')
        stats_meter = self.inference_one_epoch(0,'val')
        
        for key, value in stats_meter.items():
            print(key, value.avg)

    def _save_checkpoint(self, epoch, filename='checkpoint'):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'config': self.config,
            'best_val_loss' : self.best_loss,
            'best_val' : self.best_val,
            'best_val_metric': self.best_val_metric,
            'best_val_epoch' : self.best_val_epoch
        }
        filename = os.path.join(self.checkpoint_dir, f'{filename}.pth')
        logging.info("Saving checkpoint: {} ...".format(filename))
        torch.save(state, filename)

class RegistrationTrainer(TrainerInit):
    def __init__(self,
        config,
        data_loader,
        val_data_loader=None,
        test_data_loader=None,
        model = None,
        optimizer = None,
        scheduler = None,
        loss = None     
    ):
        if val_data_loader is not None:
            assert val_data_loader.batch_size == 1, "Val set batch size must be 1 for now."
        TrainerInit.__init__(self, config, data_loader, val_data_loader, test_data_loader, \
            model, optimizer, scheduler, loss)
        self.neg_thresh = config.neg_thresh
        self.pos_thresh = config.pos_thresh
        self.neg_weight = config.neg_weight
    
    def stats_dict(self):
        stats=dict()
        stats['circle_loss'] = 0.
        stats['feat_match_ratio'] = 0.
        # feature match recall, divided by number of ground truth pairs
        stats['rot_loss'] = 0.
        stats['trans_loss'] = 0.
        stats['total_loss'] = 0.

        return stats

    def stats_meter(self):
        meters = dict()
        stats = self.stats_dict()
        for key,_ in stats.items():
            meters[key]=AverageMeter()
        return meters

    def inference_one_batch(self, input_dict, phase):

        # Full point cloud
        src_coord = [input_dict['sinput0_C']]
        tgt_coord = [input_dict['sinput1_C']]

        src_feat = [input_dict['sinput0_F']]
        tgt_feat = [input_dict['sinput1_F']]

        src_batch_C, src_batch_F = ME.utils.sparse_collate(src_coord, src_feat)
        tgt_batch_C, tgt_batch_F = ME.utils.sparse_collate(tgt_coord, tgt_feat)
        
        transform_ab = torch.from_numpy(input_dict['T_gt'].astype(np.float32))
        eulers_ab = torch.from_numpy(input_dict['Euler_gt'])

        over_correspondences = input_dict['over_correspondences']

        inds_batch = input_dict['len_batch']
        
        #full_xyz0, full_xyz1 = input_dict['pcd0'], input_dict['pcd1']
        over_xyz0 , over_xyz1 = input_dict['pcd0_over'].to(self.device), input_dict['pcd1_over'].to(self.device)
        over_index0, over_index1 = input_dict['over_index0'].int().tolist(), input_dict['over_index1'].int().tolist()
                    
        assert phase in ['train','val','test']
        ########################################
        # training
        if (phase == 'train'):
            self.model.train()

            sinput0 = ME.SparseTensor(
                src_batch_F.to(self.device),
                coordinates=src_batch_C.to(self.device))

            sinput1 = ME.SparseTensor(
                tgt_batch_F.to(self.device),
                coordinates=tgt_batch_C.to(self.device))

            # Overlap Point Cloud
            """ src_over_coords = [input_dict['sover0_C']]
            tgt_over_coords = [input_dict['sover1_C']]

            src_over_feats = [input_dict['sover0_F']]
            tgt_over_feats = [input_dict['sover1_F']]

            src_over_batch_C, src_over_batch_F = ME.utils.sparse_collate(src_over_coords, src_over_feats)
            tgt_over_batch_C, tgt_over_batch_F = ME.utils.sparse_collate(tgt_over_coords, tgt_over_feats)

            sinput_over0 = ME.SparseTensor(
                src_over_batch_F.to(self.device),
                coordinates=src_over_batch_C.to(self.device))

            sinput_over1 = ME.SparseTensor(
                tgt_over_batch_F.to(self.device),
                coordinates=tgt_over_batch_C.to(self.device)) """

            F0, F1, rotation_ab_pred, translation_ab_pred = \
                self.model(sinput0, sinput1, over_xyz0, over_xyz1, over_index0, over_index1, inds_batch)

            identity = torch.eye(3).cuda()

            #######################################
            # loss = correspondence + transfomer
            stats = self.get_loss(over_xyz0, over_xyz1, F0, F1, over_correspondences, transform_ab.to(self.device) ) 
            rotation_ab, translation_ab = decompose_rotation_translation(transform_ab.to(self.device))

            #transform_point_cloud(over_xyz0, rotation_ab, translation_ab)

            rotation_loss= F.mse_loss(torch.matmul(rotation_ab_pred.T, rotation_ab), identity) 
            translation_loss = F.mse_loss(translation_ab_pred, translation_ab.T)
            circle_loss = stats['circle_loss']
            total_loss = rotation_loss + translation_loss + circle_loss
            
            stats['rot_loss'] = rotation_loss
            stats['trans_loss'] = translation_loss
            stats['total_loss'] = total_loss 
            
            total_loss.backward()
        
        else:
            self.model.eval()
            sinput0 = ME.SparseTensor(
                src_batch_F.to(self.device),
                coordinates=src_batch_C.to(self.device))

            sinput1 = ME.SparseTensor(
                tgt_batch_F.to(self.device),
                coordinates=tgt_batch_C.to(self.device))

            F0, F1, rotation_ab_pred, translation_ab_pred = \
                self.model(sinput0, sinput1, over_xyz0, over_xyz1, over_index0, over_index1, inds_batch )

            identity = torch.eye(3).cuda()
            
            #######################################
            # loss = correspondence + transfomer
            stats = self.get_loss(over_xyz0, over_xyz1, F0, F1, over_correspondences, transform_ab.to(self.device) )
            rotation_ab, translation_ab = decompose_rotation_translation(transform_ab.to(self.device))
            
            #transform_point_cloud(over_xyz0, rotation_ab, translation_ab)
            rotation_loss= F.mse_loss(torch.matmul(rotation_ab_pred.T, rotation_ab), identity) 
            translation_loss = F.mse_loss(translation_ab_pred, translation_ab.T)
            circle_loss = stats['circle_loss']
            total_loss = rotation_loss + translation_loss + circle_loss
            
            stats['rot_loss'] = rotation_loss
            stats['trans_loss'] = translation_loss
            stats['total_loss'] = total_loss

        stats['circle_loss'] = float(stats['circle_loss'].detach())
        stats['feat_match_ratio'] = float(stats['feat_match_ratio'].detach())
        stats['rot_loss'] = float(stats['rot_loss'].detach()) 
        stats['trans_loss'] = float(stats['trans_loss'].detach())
        stats['total_loss'] = float(stats['total_loss'].detach())
    
        return stats, rotation_ab, translation_ab, rotation_ab_pred, translation_ab_pred, eulers_ab

    def inference_one_epoch(self, epoch, phase):
        gc.collect()

        assert phase in ['train','val','test']

        stats_meter = self.stats_meter()

        if (phase=='train'):
            
            data_loader = self.data_loader
            data_loader_iter = self.data_loader.__iter__()

        elif (phase =='val'):
            data_loader = self.val_data_loader
            data_loader_iter = self.data_loader.__iter__()
            

        batch_size = self.batch_size
        num_iter = int((len(data_loader) // batch_size))

        self.optimizer.zero_grad()

        for curr_iter in tqdm(range(num_iter)):

            input_dict = data_loader_iter.next()
            inputs = dict()

            rotations_ab = []
            translations_ab = []
            rotations_ab_pred = []
            translations_ab_pred = []
            eulers_ab = []
            ##################
            # forward pass 
            # with torch.autograd.detect_anomaly():
            for i in range(self.batch_size):
                for k, v in input_dict.items():
                    inputs[k] = v[i]
            
                stats, rotation_ab, translation_ab, rotation_ab_pred, \
                    translation_ab_pred, euler_ab = self.inference_one_batch(inputs, phase)
                
                rotations_ab.append(rotation_ab.unsqueeze(dim=0).detach().cpu().numpy())
                translations_ab.append(translation_ab.unsqueeze(dim=0).detach().cpu().numpy())
                rotations_ab_pred.append(rotation_ab_pred.unsqueeze(dim=0).detach().cpu().numpy())
                translations_ab_pred.append(translation_ab_pred.unsqueeze(dim=0).detach().cpu().numpy())
                eulers_ab.append(euler_ab.unsqueeze(dim=0).numpy())

                if((curr_iter+1) % self.iter_size == 0 and phase == 'train' ):
                    gradient_valid = validate_gradient(self.model)
                    if(gradient_valid):
                        self.optimizer.step()
                    else:
                        self.logger.write('gradient not valid\n')
                    #self.optimizer.zero_grad()
            
                ################################
                # update to stats_meter
                for key,value in stats.items():
                    stats_meter[key].update(value)
            
            ##################################        
            # detach the gradients for loss terms
            rotations_ab = np.concatenate(rotations_ab, axis=0)
            translations_ab = np.concatenate(translations_ab, axis=0)
            rotations_ab_pred = np.concatenate(rotations_ab_pred, axis=0)
            translations_ab_pred = np.concatenate(translations_ab_pred, axis=0)
            eulers_ab = np.concatenate(eulers_ab, axis=0)

            rotations_ab_pred_euler = npmat2euler(rotations_ab_pred)
            r_mse_ab = np.mean((rotations_ab_pred_euler -eulers_ab) ** 2)
            r_rmse_ab = np.sqrt(r_mse_ab)
            r_mae_ab = np.mean(np.abs(rotations_ab_pred_euler - eulers_ab))
            t_mse_ab = np.mean((translations_ab - translations_ab_pred) ** 2)
            t_rmse_ab = np.sqrt(t_mse_ab)
            t_mae_ab = np.mean(np.abs(translations_ab - translations_ab_pred))

            torch.cuda.empty_cache()

            if (curr_iter + 1) % self.verbose_freq == 0 and self.verbose:
                c_iter = num_iter * (epoch - 1) + curr_iter

                for key, value in stats_meter.items():
                    self.writer.add_scalar(f'{phase}/{key}', value.avg, c_iter)
                
                message = f'{phase} Epoch: {epoch} [{curr_iter+1:4d}/{num_iter}]'
                for key,value in stats_meter.items():
                    message += f'{key}: {value.avg:.6f}\t'
      
                message += f'rot_MSE : {r_mse_ab:.4f}\t'
                message += f'rot_RMSE : {r_rmse_ab:.4f}\t'
                message += f'rot_MAE : {r_mae_ab:.4f}\t'
                message += f'trans_MSE : {t_mse_ab:.6f}\t'
                message += f'trans_RMSE : {t_rmse_ab:.6f}\t'
                message += f'trans_MAE : {t_mae_ab:.6f}\t'
                self.logger.write(message + '\n')
                logging.info(
                    "{} Epoch: {} [{:4d}/{}], ".format(phase, epoch, curr_iter+1, num_iter)+
                    "rot_MSE: {:.2f}, rot_RMSE: {:.2f}, rot_MAE: {:.2f}, trans_MSE: {:.6f}, trans_RMSE: {:.6f}, trans_MAE: {:.6f}".format(
                        r_mse_ab, r_rmse_ab, r_mae_ab, t_mse_ab, t_rmse_ab, t_mae_ab )
                )
            
        
        message = f'{phase} Epoch: {epoch}'
        for key,value in stats_meter.items():
            message += f'{key}: {value.avg:.6f}\t'

        message += f'rot_MSE : {r_mse_ab:.4f}\t'
        message += f'rot_RMSE : {r_rmse_ab:.4f}\t'
        message += f'rot_MAE : {r_mae_ab:.6f}\t'
        message += f'trans_MSE : {t_mse_ab:.6f}\t'
        message += f'trans_RMSE : {t_rmse_ab:.6f}\t'
        message += f'trans_MAE : {t_mae_ab:.6f}\t'
        self.logger.write(message+'\n')
        logging.info(
            "{} Epoch: {} , ".format(phase, epoch)+
            "rot_MSE: {:.2f}, rot_RMSE: {:.2f}, rot_MAE: {:.2f}, trans_MSE: {:.6f}, trans_RMSE: {:.6f}, trans_MAE: {:.6f}".format(
                r_mse_ab, r_rmse_ab, r_mae_ab, t_mse_ab, t_rmse_ab, t_mae_ab )
        )
        return stats_meter
