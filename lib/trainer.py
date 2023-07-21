import os
import os.path as osp
import gc
import logging
import numpy as np
import json

from rich.progress import track
import torch
torch.multiprocessing.set_sharing_strategy('file_system')

from pytorch3d.transforms.rotation_conversions import matrix_to_euler_angles

from tensorboardX import SummaryWriter
import MinkowskiEngine as ME

#from model import load_model

from model.network import PoseEstimator
from tools.file import ensure_dir
from tools.geometry import rotation_to_axis_angle
from tools.utils import Logger, rte_rre
from tools.transforms import decompose_rotation_translation
from lib.loss import transformation_loss
from lib.timer import AverageMeter

logging.getLogger().setLevel(logging.INFO)

class TrainerInit:

    def __init__(
        self,
        config,
        train_data_loader,
        val_data_loader=None,
        model = None,
        optimizer = None,
        scheduler = None
    ):
        # Model initialization

        logging.info(model)

        if config.use_gpu and not torch.cuda.is_available():
            logging.warning('Warning: There\'s no CUDA support on this machine, '
                        'training is performed on CPU.')
            raise ValueError('GPU not available, but cuda flag set')

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        self.voxel_size = config.voxel_size
        self.max_epoch = config.max_epoch
        self.batch_size = config.batch_size
        self.PoseEstimator = PoseEstimator(config).to(self.device)

        self.rte_thresh = config.rte_thresh
        self.rre_thresh = config.rre_thresh

        self.r_binsize = config.r_binsize
        self.t_binsize = config.t_binsize

        self.best_val_metric = config.best_val_metric
        self.best_val_epoch = np.inf
        self.best_val_loss = 1e5
        self.best_val_recall = -1e5
        self.start_epoch = 1
        self.checkpoint_dir = config.out_dir

        ensure_dir(self.checkpoint_dir)
        json.dump(
            config,
            open(os.path.join(self.checkpoint_dir, 'config.json'), 'w'),
            indent=4,
            sort_keys=False)

        self.iter_size = config.iter_size
        self.batch_size = train_data_loader.batch_size
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.val_batch_size = val_data_loader.batch_size

        self.test_valid = True if self.val_data_loader is not None else False
        
        self.model = self.model.to(self.device)
        self.writer = SummaryWriter(logdir=config.out_dir)
        self.logger = Logger(config.snapshot_dir)
        self.logger.write(f'#parameters {sum([x.nelement() for x in self.model.parameters()])/1000000.} M\n')

        if config.resume is not None:

            if osp.isfile(config.resume):
                logging.info("=> loading checkpoint '{}'".format(config.resume))
                state = torch.load(config.resume)

                self.start_epoch = state['epoch']
                model.load_state_dict(state['state_dict'])
                self.scheduler.load_state_dict(state['scheduler'])
                self.optimizer.load_state_dict(state['optimizer'])
                self.best_val_recall = state['best_val_recall']
                self.best_val_metric = state['best_val_metric']
                self.best_val_epoch = state['best_val_epoch']

            else:
                raise ValueError(f"=> no checkpoint found at '{config.resume}'")

    def train(self):
        """
        Full training logic
        """
        # Baseline random feature performance
        """ if self.test_valid:
            with torch.no_grad():
                val_dict = self.inference_one_epoch('val')
            for k, v in val_dict.items():
                self.writer.add_scalar(f'val/{k}', v.avg, 0 ) """
                
        for epoch in range(self.start_epoch, self.max_epoch + 1):

            lr = self.scheduler.get_lr()
            logging.info(f" Epoch: {epoch}, LR: {lr}")
            with torch.autograd.set_detect_anomaly(True):
                train_dict, train_rotations_ab, train_translations_ab, train_rotations_ab_pred, \
                    train_translations_ab_pred, train_eulers_ab = self.inference_one_epoch('train')

            self._save_checkpoint(epoch)
            self.scheduler.step()
    
            with torch.no_grad():
                val_dict, val_rotations_ab, val_translations_ab, val_rotations_ab_pred, \
                    val_translations_ab_pred, val_eulers_ab = self.inference_one_epoch('val')

            train_loss = train_dict['loss'].avg
            train_rre = train_dict['rre'].avg
            train_rte = train_dict['rte'].avg

            val_rre = val_dict['rre'].avg
            val_rte = val_dict['rte'].avg
            
            train_rotations_ab_pred_euler = matrix_to_euler_angles(train_rotations_ab_pred, "ZYX")
            train_rotations_ab_pred_euler = train_rotations_ab_pred_euler.detach().cpu().numpy()
            train_r_mse_ab = np.mean((train_rotations_ab_pred_euler - np.degrees(train_eulers_ab)) ** 2)
            train_r_rmse_ab = np.sqrt(train_r_mse_ab)
            train_r_mae_ab = np.mean(np.abs(train_rotations_ab_pred_euler - np.degrees(train_eulers_ab)))
            train_t_mse_ab = np.mean((train_translations_ab - train_translations_ab_pred) ** 2)
            train_t_rmse_ab = np.sqrt(train_t_mse_ab)
            train_t_mae_ab = np.mean(np.abs(train_translations_ab - train_translations_ab_pred))

            val_rotations_ab_pred_euler = matrix_to_euler_angles(val_rotations_ab_pred, "ZYX")
            val_rotations_ab_pred_euler = val_rotations_ab_pred_euler.detach().cpu().numpy()
            val_r_mse_ab = np.mean((val_rotations_ab_pred_euler - np.degrees(val_eulers_ab)) ** 2)
            val_r_rmse_ab = np.sqrt(val_r_mse_ab)
            val_r_mae_ab = np.mean(np.abs(val_rotations_ab_pred_euler - np.degrees(val_eulers_ab)))
            val_t_mse_ab = np.mean((val_translations_ab - val_translations_ab_pred) ** 2)
            val_t_rmse_ab = np.sqrt(val_t_mse_ab)
            val_t_mae_ab = np.mean(np.abs(val_translations_ab - val_translations_ab_pred))

            ##########################################################
            # train write

            message = f'Train Epoch: {epoch} '
            for key,value in train_dict.items():
                self.writer.add_scalar(f'Train/{key}', value.avg, epoch)
                message += f'{key}: {value.avg:.6f}\t'

            message += f'rot_MSE : {train_r_mse_ab:.4f}\t'
            message += f'rot_RMSE : {train_r_rmse_ab:.4f}\t'
            message += f'rot_MAE : {train_r_mae_ab:.6f}\t'
            message += f'trans_MSE : {train_t_mse_ab:.6f}\t'
            message += f'trans_RMSE : {train_t_rmse_ab:.6f}\t'
            message += f'trans_MAE : {train_t_mae_ab:.6f}\t'

            self.logger.write(message+'\n')

            logging.info(
                "Train Epoch: {} , ".format(epoch) +
                "Loss : {:.6f}, rre: {:.6f}, rte: {:.6f}, rot_MSE: {:.6f}, rot_RMSE: {:.6f}, rot_MAE: {:.6f}, trans_MSE: {:.6f}, trans_RMSE: {:.6f}, trans_MAE: {:.6f}".format(
                    train_loss, train_rre, train_rte, train_r_mse_ab, train_r_rmse_ab, train_r_mae_ab, train_t_mse_ab, train_t_rmse_ab, train_t_mae_ab )
            )

            ##########################################################
            # Validation write

            message = f'Validation Epoch: {epoch} '
            for key,value in val_dict.items():
                self.writer.add_scalar(f'Validation/{key}', value.avg, epoch)
                message += f'{key}: {value.avg:.6f}\t'

            message += f'rot_MSE : {val_r_mse_ab:.4f}\t'
            message += f'rot_RMSE : {val_r_rmse_ab:.4f}\t'
            message += f'rot_MAE : {val_r_mae_ab:.6f}\t'
            message += f'trans_MSE : {val_t_mse_ab:.6f}\t'
            message += f'trans_RMSE : {val_t_rmse_ab:.6f}\t'
            message += f'trans_MAE : {val_t_mae_ab:.6f}\t'

            self.logger.write(message+'\n')

            logging.info(
                "Validation Epoch: {} , ".format(epoch)+
                "rre: {:.6f}, rte: {:.6f}, rot_MSE: {:.6f}, rot_RMSE: {:.6f}, rot_MAE: {:.6f}, trans_MSE: {:.6f}, trans_RMSE: {:.6f}, trans_MAE: {:.6f}".format(
                    val_rre, val_rte, val_r_mse_ab, val_r_rmse_ab, val_r_mae_ab, val_t_mse_ab, val_t_rmse_ab, val_t_mae_ab )
            )
            
            if val_dict[self.best_val_metric].avg >= self.best_val_recall :
                logging.info(
                    f'Saving the best val model with {self.best_val_metric}: {val_dict[self.best_val_metric].avg}'
                )
                self.best_val_recall = val_dict[self.best_val_metric].avg
                self.best_val_epoch = epoch
                self._save_checkpoint(epoch, 'best_val_recall_checkpoint')

            else:
                logging.info(
                    f'Current best val model with {self.best_val_metric}: {self.best_val_recall} at epoch {self.best_val_epoch}'
                )

    def _save_checkpoint(self, epoch, filename='checkpoint'):

        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'config': self.config,
            'best_val_recall' : self.best_val_recall,
            'best_val_metric': self.best_val_metric,
            'best_val_epoch' : self.best_val_epoch
        }

        filename = os.path.join(self.checkpoint_dir, f'{filename}.pth')
        logging.info("Saving checkpoint: {} ...".format(filename))
        torch.save(state, filename)

class RegistrationTrainer(TrainerInit):

    def __init__(self,
        config,
        train_data_loader,
        val_data_loader=None,
        model = None,
        optimizer = None,
        scheduler = None,
    ):
        if val_data_loader is not None:
            assert val_data_loader.batch_size == 1, "Val set batch size must be 1 for now."
        TrainerInit.__init__(self, config, train_data_loader, val_data_loader, \
            model, optimizer, scheduler)

        self.neg_thresh = config.neg_thresh
        self.pos_thresh = config.pos_thresh

        self.alpha = config.alpha
        self.beta = config.beta

    def stats_dict(self, phase):

        stats=dict()

        if (phase == "train"):
            stats['loss'] = 0.
            stats['recall'] = 0.
            stats['rte'] = 0.
            stats['rre'] = 0.

        elif(phase == "val"):
            stats['recall'] = 0.
            stats['rte'] = 0.
            stats['rre'] = 0.

        return stats

    def stats_meter(self, phase):

        meters = dict()
        stats = self.stats_dict(phase)

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
        eulers_ab = input_dict['Euler_gt']

        over_xyz0 , over_xyz1 = input_dict['pcd0_over'].to(self.device), input_dict['pcd1_over'].to(self.device)
        over_index0, over_index1 = input_dict['over_index0'].int().tolist(), input_dict['over_index1'].int().tolist()
      
        assert phase in ['train','val','test']

        ########################################
        # training
        
        if (phase == 'train'):
            gc.collect()
            self.model.train()
            self.optimizer.zero_grad()

            sinput0 = ME.SparseTensor(
                src_batch_F.to(self.device),
                coordinates=src_batch_C.to(self.device))

            sinput1 = ME.SparseTensor(
                tgt_batch_F.to(self.device),
                coordinates=tgt_batch_C.to(self.device))

            Transform_pred, votes = self.PoseEstimator(sinput0, sinput1, over_xyz0, over_xyz1, over_index0, over_index1, self.model)

            rotation_ab, translation_ab = decompose_rotation_translation(transform_ab.to(self.device))
            rotation_pred, translation_pred = decompose_rotation_translation(Transform_pred.to(self.device))

            # Hough Voting
            hspace = votes
            del votes

            angle_gt = rotation_to_axis_angle(transform_ab[:3,:3])
            index_gt = torch.cat(
                [
                    torch.floor(angle_gt / self.r_binsize),
                    torch.floor(transform_ab[:3, 3] / self.t_binsize),
                ],
                dim=0,
            ).int()

            index_diff = (hspace.C[:, 1:] - index_gt.to(self.device)).abs().sum(dim=1)
            index_min = index_diff.argmin()

            # if the offset is larger than 3 voxels, skip current batch
            """ if index_diff[index_min].item() > 3:
                return """

            target = torch.zeros_like(hspace.F)
            target[index_min] = 1.0

            # criteria = BalancedLoss()
            criteria = torch.nn.BCEWithLogitsLoss()
            vote_loss = criteria(hspace.F, target)
            
            rotation_loss, translation_loss = transformation_loss(rotation_ab, translation_ab, rotation_pred, translation_pred)
            
            loss = (vote_loss * self.alpha) + (rotation_loss  + translation_loss) * self.beta

            loss_float = loss.detach().cpu().item()

            success, rte, rre = rte_rre(
                Transform_pred.cpu().numpy(), transform_ab.cpu().numpy(), self.rte_thresh, self.rre_thresh
            )

            values = dict(
                loss=loss_float, recall=success, rte=rte, rre=rre
            )
            
            loss.requires_grad_(True)
            loss.backward()
            self.optimizer.step()

            gc.collect()
            torch.cuda.empty_cache()
        
        else:
            with torch.no_grad():
                self.model.eval()
                sinput0 = ME.SparseTensor(
                    src_batch_F.to(self.device),
                    coordinates=src_batch_C.to(self.device))

                sinput1 = ME.SparseTensor(
                    tgt_batch_F.to(self.device),
                    coordinates=tgt_batch_C.to(self.device))

                Transform_pred, votes = \
                    self.PoseEstimator(sinput0, sinput1, over_xyz0, over_xyz1, over_index0, over_index1, self.model )
                
                rotation_ab, translation_ab = decompose_rotation_translation(transform_ab.to(self.device))
                rotation_pred, translation_pred = decompose_rotation_translation(Transform_pred.to(self.device))

                success, rte, rre = rte_rre(
                    Transform_pred.cpu().numpy(), transform_ab.cpu().numpy(), self.rte_thresh, self.rre_thresh
                )

                values = dict(
                    recall=success, rte=rte, rre=rre
                )

        return values, rotation_ab, translation_ab, rotation_pred, translation_pred, eulers_ab

    def inference_one_epoch(self, phase):
        gc.collect()

        rotations_ab = []
        translations_ab = []
        rotations_ab_pred = []
        translations_ab_pred = []
        eulers_ab = []

        assert phase in ['train','val','test']

        stats_meter = self.stats_meter(phase)

        if (phase=='train'):
            data_loader = self.train_data_loader
            data_loader_iter = self.train_data_loader.__iter__()
            batch_size = self.batch_size
            
        elif (phase =='val'):
            data_loader = self.val_data_loader
            data_loader_iter = self.val_data_loader.__iter__()
            batch_size = self.val_batch_size

        num_iter = int((len(data_loader) // batch_size))

        for curr_iter in track(range(num_iter)):

            input_dict = data_loader_iter.next()
            inputs = dict()
            stats = dict()

            total_loss, total_recall, total_rte, total_rre = 0, 0, 0, 0

            ##################
            # forward pass 
            # with torch.autograd.detect_anomaly():
            for i in range(batch_size):
                for k, v in input_dict.items():
                    inputs[k] = v[i]
            
                stats, rotation_ab, translation_ab, rotation_ab_pred, \
                    translation_ab_pred, euler_ab = self.inference_one_batch(inputs, phase)
                
                rotations_ab.append(rotation_ab.unsqueeze(dim=0).detach().cpu().numpy())
                translations_ab.append(translation_ab.unsqueeze(dim=0).detach().cpu().numpy())
                rotations_ab_pred.append(rotation_ab_pred.unsqueeze(dim=0))
                translations_ab_pred.append(translation_ab_pred.unsqueeze(dim=0).detach().cpu().numpy())
                eulers_ab.append(euler_ab.unsqueeze(dim=0).numpy())

                ################################
                # update to stats_meter
                if (phase == 'train'):
                    loss = stats['loss']

                recall = stats['recall']
                rte = stats['rte']
                rre = stats['rre']

                if (phase == 'train'):
                    total_loss += loss

                total_recall += recall
                total_rte += rte
                total_rre += rre

            if (phase == 'train'):
                total_loss /= batch_size
            
            total_recall /= batch_size
            total_rte /= batch_size
            total_rre /= batch_size

            if (phase == 'train'):
                stats['loss'] = total_loss

            stats['recall'] = total_recall
            stats['rte'] = total_rte
            stats['rre'] = total_rre

            for key,value in stats.items():
                stats_meter[key].update(value)
            
        ##################################        
        # detach the gradients for loss terms
        rotations_ab = np.concatenate(rotations_ab, axis=0)
        translations_ab = np.concatenate(translations_ab, axis=0)
        rotations_ab_pred = torch.cat(rotations_ab_pred, axis=0)
        translations_ab_pred = np.concatenate(translations_ab_pred, axis=0)
        eulers_ab = np.concatenate(eulers_ab, axis=0)

        return stats_meter, rotations_ab, translations_ab, rotations_ab_pred, translations_ab_pred, eulers_ab
