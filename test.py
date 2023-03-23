import argparse
import logging
import os
import time
import torch
import numpy as np
from easydict import EasyDict as edict

from rich.console import Console
from rich.progress import track
from rich.table import Table

from config_test import get_config
from datasets.threedmatch_dataset import ThreeDMatchTestDataset
from datasets.data_loaders import make_data_loader

from model.network import PoseEstimator
from model.simpleunet import SimpleNet
from lib.timer import Timer, AverageMeter

from tools.file import ensure_dir
from tools.test_utils import datasets_setting
from tools.utils import rte_rre


os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

ALL_DATASETS = [ThreeDMatchTestDataset]
dataset_str_mapping = {d.__name__: d for d in ALL_DATASETS}

def print_table(subset_names, stats, rte_ths, rre_ths):
    console = Console()
    table = Table(show_header=True, header_style="bold")

    columns = ["scene", "recall", "rte", "rre", "time"]
    for col in columns:
        table.add_column(col)

    if stats.ndim == 3:
        stats = stats[-1, :, :]

    stats[:, 0] = (stats[:, 1] < rte_ths) * (stats[:, 2] < rre_ths)
    scene_vals = np.zeros((len(subset_names), 4))
    for sid, _ in enumerate(subset_names):
        curr_scene = stats[:, -1] == sid
        if curr_scene.sum() > 0:
            curr_scene_stats = stats[curr_scene]
            success = curr_scene_stats[:, 0] > 0
            recall = success.mean()
            scene_vals[sid][0] = recall
            scene_vals[sid][1:4] = curr_scene_stats[success, 1:4].mean(0)
        else:
            scene_vals[sid] = None

    for sid, vals in zip(subset_names, scene_vals):
        table.add_row(sid, *[f"{v:.4f}" for v in vals])

    success = stats[:, 0] > 0
    recall = success.mean()
    metrics = stats[success, :4].mean(0)
    metrics[0] = recall
    table.add_row("avg", *[f"{m:.4f}" for m in metrics])
    console.print(table)

def run_benchmark(
    data_loader,
    method,
    TE_THRESH,
    RE_THRESH,
    log_interval=100,
    device = None,
    voxel_size = None,
    overlap = None
):
    tot_num_data = len(data_loader)
    data_loader_iter = data_loader.__iter__()

    dataset = data_loader.dataset
    subset_names = dataset.subset_names

    pose_estimate = PoseEstimator(config).to(device)

    stats = np.zeros((tot_num_data, 5))
    stats[:, -1] = -1
    poses = []

    with torch.no_grad():
        for batch_idx in track(range(tot_num_data)):
            batch = data_loader_iter.next()
            sname, xyz0, xyz1, trans, f0, f1 = batch[0]
            sid = subset_names.index(sname)
            T_gt = np.linalg.inv(trans)
            #T_gt = trans

            sinput0, sinput1, src_over, tgt_over, over_index0, over_index1 = datasets_setting(xyz0, xyz1, T_gt, voxel_size, overlap, device)

            start = time.time()            
            T, _, _ = pose_estimate(sinput0, sinput1, torch.from_numpy(src_over).to(device), torch.from_numpy(tgt_over).to(device), over_index0, over_index1, method.to(device))
            end = time.time()

            result = rte_rre(T, T_gt, TE_THRESH, RE_THRESH)
            stats[batch_idx, :3] = result
            stats[batch_idx, 3] = end - start
            stats[batch_idx, 4] = sid
            poses.append(T.numpy())

            recall = str(round(result[0],2))
            rte = str(round(result[1],2))
            rre = str(round(result[2],2))

            filename0 = f0.split('/')[-1]
            filename1 = f1.split('/')[-1]

            if float(recall) == 1 and float(rte) < 1.5 and float(rre) < 1 :
                with open("./data_list.txt", "a") as f:
                    f.write("filename0 : " + sname + "/" + filename0 + '\n' 
                        "filename1 : " + sname + "/" + filename1 + '\n'
                        + "recall : " + recall + " RTE : " + rte +  " RRE :" + rre + '\n')
           
            if batch_idx % log_interval == 0 and batch_idx > 0:
                cur_stats = stats[:batch_idx]
                cur_recall = cur_stats[:, 0].mean() * 100
                cur_rte = cur_stats[cur_stats[:, 0] > 0, 1].mean()
                cur_rre = cur_stats[cur_stats[:, 0] > 0, 2].mean()
                print(
                    f"recall: {cur_recall:.2f}, rte: {cur_rte:.2f}, rre: {cur_rre:.2f}"
                )

    return subset_names, stats, np.stack(poses, axis=0)

def test(args):

    ensure_dir(args.target)
    checkpoint = torch.load(args.model,map_location='cpu')
    config = checkpoint['config']

    # initialize Model
    search_voxel_size = config.voxel_size * config.positive_pair_search_voxel_size_multiplier
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SimpleNet(
            conv1_kernel_size=config.conv1_kernel_size,
            D=6)
            
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
  
    # initialize data loader
    test_loader = make_data_loader(
        args,
        args.test_phase,
        args.val_batch_size,
        num_threads=args.val_num_thread)

    TE_THRESH = test_loader.dataset.TE_THRESH
    RE_THRESH = test_loader.dataset.RE_THRESH

    subset_names, stats, poses = run_benchmark(
        test_loader,
        method = model,
        TE_THRESH = TE_THRESH,
        RE_THRESH = RE_THRESH,
        log_interval=100,
        device = device,
        voxel_size = config.voxel_size,
        overlap = search_voxel_size
    )

    # print_table
    print_table(subset_names, stats, TE_THRESH, RE_THRESH)

    # save results
    exp_dir = os.path.join(args.out_dir, args.run_name)
    ensure_dir(exp_dir)
    stat_filename = os.path.join(exp_dir, "stats.npz")
    conf_filename = os.path.join(exp_dir, "config.txt")
    np.savez(stat_filename, stats=stats, names=["dhvr"], poses=poses)
    with open(conf_filename, "w") as f:
        f.write(gin.operative_config_str())
    logging.info(f"Saved results to {stat_filename}, {conf_filename}")



if __name__ == "__main__":

    config = get_config()

    dconfig = vars(config)
    config = edict(dconfig)
    # start test
    test(config)
