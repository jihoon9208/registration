import os
import os.path as osp
import gc
import logging
import numpy as np
import json
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd.grad_mode import no_grad
from tensorboardX import SummaryWriter
import MinkowskiEngine as ME
from sklearn.neighbors import NearestNeighbors

#from model import load_model
import model
from tools.file import ensure_dir
from tools.utils import validate_gradient
from tools.model_util import npmat2euler, transform_point_cloud, cdist_torch
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
        self.best_feat_recall = -1e5

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

        self.get_loss = loss

        if config.resume is not None:
            if osp.isfile(config.resume):
                logging.info("=> loading checkpoint '{}'".format(config.resume))
                state = torch.load(config.resume)
                self.start_epoch = state['epoch']
                model.load_state_dict(state['state_dict'])
                self.scheduler.load_state_dict(state['scheduler'])
                self.optimizer.load_state_dict(state['optimizer'])
                self.best_loss = state['best_loss']
                self.best_recall = state['best_recall']
            else:
                raise ValueError(f"=> no checkpoint found at '{config.resume}'")

    def train(self):
        """
        Full training logic
        """
        # Baseline random feature performance
        """ if self.test_valid:
            with torch.no_grad():
                val_dict = self._valid_epoch()

            for k, v in val_dict.items():
                self.writer.add_scalar(f'val/{k}', v, 0 ) """
        
        for epoch in range(self.start_epoch, self.max_epoch + 1):
            lr = self.scheduler.get_lr()
            logging.info(f" Epoch: {epoch}, LR: {lr}")
            #with torch.autograd.set_detect_anomaly(True):
            self.inference_one_epoch(epoch, 'train')
            self._save_checkpoint(epoch)
            self.scheduler.step()
            
            if self.test_valid and epoch % self.verbose_freq == 0:
                with torch.no_grad():
                    val_dict = self.inference_one_epoch(epoch, 'val')

                for k, v in val_dict.items():
                    self.writer.add_scalar(f'val/{k}', v, epoch)
                if val_dict['circle_loss'].avg < self.best_loss:
                    self.best_loss = val_dict['circle_loss'].avg
                    self._snapshot(epoch,'best_loss')
                if val_dict['recall'].avg > self.best_recall:
                    self.best_recall = val_dict['recall'].avg
                    self._snapshot(epoch,'best_recall')
                else:
                    logging.info(
                        f'Current best val model with {self.best_val_metric}: {self.best_val} at epoch {self.best_val_epoch}'
                    )

    def _save_checkpoint(self, epoch, filename='checkpoint'):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'config': self.config,
            'best_loss' : self.best_loss,
            'best_feat_recall' : self.best_feat_recall,
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
    
    def stata_dict(self):
        stata=dict()
        stata['circle_loss'] = 0.
        stata['feat_recall'] = 0.  # feature match recall, divided by number of ground truth pairs
        stata['rota_loss'] = 0.
        stata['trans_loss'] = 0.

        return stata

    def stata_meter(self):
        meters=dict()
        stata=self.stata_dict()
        for key,_ in stata.items():
            meters[key]=AverageMeter()
        return meters

    """ def apply_transform(self, pts, trans):
        R = trans[:3, :3]
        T = trans[:3, 3]
        return pts @ R.t() + T

    def ground_truth_attention(self, p1, p2, trans):
        
        p1 = sample_points(p1, self.num_points)
        p2 = sample_points(p2, self.num_points)

        ideal_pts2 = self.apply_transform(p1, trans)    

        nn = NearestNeighbors(n_neighbors=1).fit(p2)
        distance, neighbors = nn.kneighbors(ideal_pts2)
        neighbors1 = neighbors[distance < 0.3]
        pcd1 = p2[neighbors1]
        
        # Search NN for each p2 in ideal_pt2
        nn = NearestNeighbors(n_neighbors=1).fit(ideal_pts2)
        distance, neighbors = nn.kneighbors(p2)
        neighbors2 = neighbors[distance < 0.3]
        pcd0 = p1[neighbors2]
    
        return pcd0, pcd1, neighbors2, neighbors1 """

    def inference_one_batch(self, input_dict, epoch, phase):

        rotations_ab = []
        translations_ab = []
        rotations_ab_pred = []
        translations_ab_pred = []
        eulers_ab = []

        
        assert phase in ['train','val','test']
        ########################################
        # training
        if (phase == 'train'):
            self.model.train()
            sinput0 = ME.SparseTensor(
                        input_dict['sinput0_F'].to(self.device),
                        coordinates=input_dict['sinput0_C'].to(self.device))

            sinput1 = ME.SparseTensor(
                input_dict['sinput1_F'].to(self.device),
                coordinates=input_dict['sinput1_C'].to(self.device))

            pcd0_xyz = input_dict['pcd0'].to(self.device)
            pcd1_xyz = input_dict['pcd1'].to(self.device)
            
            rotation_ab = input_dict['rot'].to(self.device)
            translation_ab = input_dict['trans'].to(self.device)
            eulers_ab = input_dict['Euler_gt']
            
            pos_pairs = input_dict['correspondences'].long().to(self.device)
            
            over_xyz0 , over_xyz1 = input_dict['pcd0_over'], input_dict['pcd1_over']
            over_index0, over_index1 = input_dict['over_index0'].to(self.device), input_dict['over_index1'].to(self.device)
            
            F0, F1, rotation_ab_pred, translation_ab_pred, pred_corr = self.model(sinput0, sinput1, \
                over_xyz0 , over_xyz1, over_index0, over_index1).to(self.device)

            ## save rotation and translation
            rotations_ab.append(rotation_ab.detach().cpu().numpy())
            translations_ab.append(translation_ab.detach().cpu().numpy())
            rotations_ab_pred.append(rotation_ab_pred.detach().cpu().numpy())
            translations_ab_pred.append(translation_ab_pred.detach().cpu().numpy())
            eulers_ab.append(eulers_ab.numpy())

            identity = torch.eye(3).cuda().unsqueeze(0).repeat(self.batch_size, 1, 1)
            
            #######################################
            # loss = correspondence + transfomer
            stats = self.get_loss(pcd0_xyz, pcd1_xyz, F0, F1, pos_pairs, rotation_ab, translation_ab )
            ind_mask = (cdist_torch(transform_point_cloud(pcd0_xyz, rotation_ab, translation_ab), pcd1_xyz, points_dim=3).min(dim=2).values < 0.05)
            
            stats['corr_loss'] = 0.95**epoch * ((pred_corr - transform_point_cloud(pcd0_xyz, rotation_ab, translation_ab)) ** 2).sum(dim=1).view(-1)[ind_mask.view(-1)].mean()
            stats['rota_loss'] = F.mse_loss(torch.matmul(rotation_ab_pred.transpose(2, 1), rotation_ab), identity) 
            stats['trans_loss'] = F.mse_loss(translation_ab_pred, translation_ab)

            total_loss = stats['circle_loss'] + stats['corr_loss'] + stats['rota_loss'] + stats['trans_loss']
            total_loss.backward()
        
        else:
            self.model.eval()
            sinput0 = ME.SparseTensor(
                        input_dict['sinput0_F'].to(self.device),
                        coordinates=input_dict['sinput0_C'].to(self.device))

            sinput1 = ME.SparseTensor(
                input_dict['sinput1_F'].to(self.device),
                coordinates=input_dict['sinput1_C'].to(self.device))

            pcd0_xyz = input_dict['pcd0'].to(self.device)
            pcd1_xyz = input_dict['pcd1'].to(self.device)

            rotation_ab = input_dict['rot'].to(self.device)
            translation_ab = input_dict['trans'].to(self.device)
            eulers_ab = input_dict['Euler_gt'].to(self.device)

            pos_pairs = input_dict['correspondences'].long().to(self.device)
            
            over_xyz0 , over_xyz1 = input_dict['pcd0_over'].to(self.device), input_dict['pcd1_over'].to(self.device)
            over_index0, over_index1 = input_dict['over_index0'].to(self.device), input_dict['over_index1'].to(self.device)
            
            F0, F1, rotation_ab_pred, translation_ab_pred, pred_corr = self.model(sinput0, sinput1, \
                over_xyz0, over_xyz1, over_index0, over_index1).to(self.device)

            ## save rotation and translation
            rotations_ab.append(rotation_ab.detach().cpu().numpy())
            translations_ab.append(translation_ab.detach().cpu().numpy())
            rotations_ab_pred.append(rotation_ab_pred.detach().cpu().numpy())
            translations_ab_pred.append(translation_ab_pred.detach().cpu().numpy())
            eulers_ab.append(eulers_ab.numpy())

            identity = torch.eye(3).cuda().unsqueeze(0).repeat(self.batch_size, 1, 1)
            
            #######################################
            # loss = correspondence + transfomer
            data = self.get_loss(pcd0_xyz, pcd1_xyz, F0, F1, pos_pairs, rotation_ab, translation_ab )
            ind_mask = (cdist_torch(transform_point_cloud(pcd0_xyz, rotation_ab, translation_ab), pcd1_xyz, points_dim=3).min(dim=2).values < 0.05)

            stats['corr_loss'] = 0.95**epoch * ((pred_corr - transform_point_cloud(pcd0_xyz, rotation_ab, translation_ab)) ** 2).sum(dim=1).view(-1)[ind_mask.view(-1)].mean()
            stats['rota_loss'] = F.mse_loss(torch.matmul(rotation_ab_pred.transpose(2, 1), rotation_ab), identity) 
            stats['trans_loss'] = F.mse_loss(translation_ab_pred, translation_ab)

        ##################################        
        # detach the gradients for loss terms
        rotations_ab = np.concatenate(rotations_ab, axis=0)
        translations_ab = np.concatenate(translations_ab, axis=0)
        rotations_ab_pred = np.concatenate(rotations_ab_pred, axis=0)
        translations_ab_pred = np.concatenate(translations_ab_pred, axis=0)
        eulers_ab = np.concatenate(eulers_ab, axis=0)

        stats['circle_loss'] = float(stats['circle_loss'].detach())
        stats['corr_loss'] = float(stats['corr_loss']).detach()
        stats['rota_loss'] = float(stats['rota_loss'].detach()) 
        stats['trans_loss'] = float(stats['trans_loss'].detach())
    
        return stats, rotations_ab, translations_ab, rotations_ab_pred, translations_ab_pred, eulers_ab

    def inference_one_epoch(self, epoch, phase):
        gc.collect()

        assert phase in ['train','val','test']

        stats_meter = AverageMeter()
        data_timer, total_timer = Timer(), Timer()

        batch_loss = 0
        data_time = 0

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
           
            try :
                ##################
                # forward pass
                # with torch.autograd.detect_anomaly():
                stats, rotations_ab, translations_ab, rotations_ab_pred, \
                    translations_ab_pred, eulers_ab = self.inference_one_batch(input_dict, epoch, phase)

                train_rotations_ab_pred_euler = npmat2euler(rotations_ab_pred)
                train_r_mse_ab = np.mean((train_rotations_ab_pred_euler - np.degrees(eulers_ab)) ** 2)
                train_r_rmse_ab = np.sqrt(train_r_mse_ab)
                train_r_mae_ab = np.mean(np.abs(train_rotations_ab_pred_euler - np.degrees(eulers_ab)))
                train_t_mse_ab = np.mean((translations_ab - translations_ab_pred) ** 2)
                train_t_rmse_ab = np.sqrt(train_t_mse_ab)
                train_t_mae_ab = np.mean(np.abs(translations_ab - translations_ab_pred))

                if((curr_iter+1) % self.iter_size == 0 and phase == 'train' ):
                    gradient_valid = validate_gradient(self.model)
                    if(gradient_valid):
                        self.optimizer.step()
                    else:
                        self.logger.write('gradient not valid\n')
                    self.optimizer.zero_grad()
                
                ################################
                # update to stats_meter

                for key,value in stats.items():
                    stats_meter[key].update(value)

            except RuntimeError as inst :
                pass
        
            torch.cuda.empty_cache()

            if (curr_iter + 1) % self.verbose_freq == 0 and self.verbose:
                curr_iter = num_iter * (epoch - 1) + curr_iter
                for key, value in stats_meter.items():
                    self.writer.add_scalar(f'{phase}/{key}', value.avg, curr_iter)
                
                message = f'{phase} Epoch: {epoch} [{curr_iter+1:4d}/{num_iter}]'
                for key,value in stats_meter.items():
                    message += f'{key}: {value.avg:.2f}\t'
                message += f'rot_MSE : {train_r_mse_ab:.2f}\t'
                message += f'rot_RMSE : {train_r_rmse_ab:.2f}\t'
                message += f'rot_MAE : {train_r_mae_ab:.2f}\t'
                message += f'trans_MSE : {train_t_mse_ab:.2f}\t'
                message += f'trans_RMSE : {train_t_rmse_ab:.2f}\t'
                message += f'trans_MAE : {train_t_mae_ab:.2f}\t'
                self.logger.write(message + '\n')

            total_timer.reset()
        
        message = f'{phase} Epoch: {epoch}'
        for key,value in stats_meter.items():
            message += f'{key}: {value.avg:.2f}\t'
        message += f'rot_MSE : {train_r_mse_ab:.2f}\t'
        message += f'rot_RMSE : {train_r_rmse_ab:.2f}\t'
        message += f'rot_MAE : {train_r_mae_ab:.2f}\t'
        message += f'trans_MSE : {train_t_mse_ab:.2f}\t'
        message += f'trans_RMSE : {train_t_rmse_ab:.2f}\t'
        message += f'trans_MAE : {train_t_mae_ab:.2f}\t'
        self.logger.write(message+'\n')

        return stats_meter
