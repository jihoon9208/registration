# -*- coding: future_fstrings -*-
from turtle import clear
import open3d as o3d  # prevent loading error
import os
import sys
import json
import logging
import torch
from easydict import EasyDict as edict
from model import load_model
from model.network import PoseEstimator
from model.simpleunet import SimpleNet

from datasets.data_loaders import make_data_loader
from config import get_config
import torch.optim as optim
from datasets.collate import CollateFunc as coll
from lib.trainer import RegistrationTrainer
from lib.loss import MetricLoss

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
torch.multiprocessing.set_sharing_strategy('file_system')


def get_trainer(trainer):
    if trainer == 'RegistrationTrainer':
        return RegistrationTrainer

    else:
        raise ValueError(f'Trainer {trainer} not found')


def main(config, resume=False):

    # Model initialization
    
    model = SimpleNet(
            conv1_kernel_size=config.conv1_kernel_size,
            D=6)

    #model = PoseEstimator(config)

    if config.optimizer == 'SGD':
        optimizer = getattr(optim, config.optimizer)(
                model.parameters(),
                lr=config.lr,
                momentum=config.momentum,
                weight_decay=config.weight_decay)

    if config.optimizer == 'Adam':
        optimizer = getattr(optim, config.optimizer)(
                model.parameters(),
                lr=config.lr,
                betas=(0.9, 0.999),
                weight_decay=config.weight_decay)

    if config.optimizer == 'AdamW':
        optimizer = getattr(optim, config.optimizer)(
                model.parameters(),
                lr=config.lr,
                betas=(0.9, 0.999),
                weight_decay=config.weight_decay)

    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.exp_gamma)
    

    #Predator dataloader 
    """ train_set, val_set, benchmark_set = get_datasets(config)

    train_loader = torch.utils.data.DataLoader(train_set, 
                                        batch_size=config.batch_size, 
                                        shuffle=True,
                                        num_workers=config.train_num_thread,
                                        collate_fn=coll.collate_pair_fn,
                                        pin_memory=False,
                                        drop_last=False)
    val_loader = torch.utils.data.DataLoader(val_set, 
                                        batch_size=config.val_batch_size, 
                                        shuffle=True,
                                        num_workers=config.val_num_thread,
                                        collate_fn=coll.collate_pair_fn,
                                        pin_memory=False,
                                        drop_last=False)
    test_loader = torch.utils.data.DataLoader(benchmark_set, 
                                        batch_size=config.batch_size, 
                                        shuffle=False,
                                        num_workers=config.test_num_thread,
                                        collate_fn=coll.collate_pair_fn,
                                        pin_memory=False,
                                        drop_last=False) """

    train_loader = make_data_loader(
        config,
        config.train_phase,
        config.batch_size,
        num_threads=config.train_num_thread)

    val_loader = make_data_loader(
        config,
        config.val_phase,
        config.val_batch_size,
        num_threads=config.val_num_thread)

    test_loader = make_data_loader(
        config,
        config.val_phase,
        config.val_batch_size,
        num_threads=config.val_num_thread)

    get_loss = MetricLoss(config)

    Trainer = get_trainer(config.trainer)

    trainer = Trainer(
        config=config,
        data_loader=train_loader,
        val_data_loader=val_loader,
        test_data_loader=test_loader,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss=get_loss
    )

    trainer.train()

if __name__ == "__main__":
    logger = logging.getLogger()
    config = get_config()

    dconfig = vars(config)
    if config.resume_dir:
        resume_config = json.load(open(config.resume_dir + '/config.json', 'r'))
        for k in dconfig:
            if k not in ['resume_dir'] and k in resume_config:
                dconfig[k] = resume_config[k]
        dconfig['resume'] = resume_config['out_dir'] + '/checkpoint.pth'

    logging.info('===> Configurations')
    for k in dconfig:
        logging.info('    {}: {}'.format(k, dconfig[k]))

    # Convert to dict
    config = edict(dconfig)
    main(config)
