import argparse

arg_lists = []
parser = argparse.ArgumentParser()


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


def str2bool(v):
    return v.lower() in ('true', '1')

logging_arg = add_argument_group('Logging')
logging_arg.add_argument('--out_dir', type=str, default='outputs')
logging_arg.add_argument('--snapshot_dir', type=str, default='outputs/snapshot')

trainer_arg = add_argument_group('Trainer')
trainer_arg.add_argument('--trainer', type=str, default='RegistrationTrainer')
trainer_arg.add_argument('--batch_size', type=int, default=8)
trainer_arg.add_argument('--val_batch_size', type=int, default=1)

# Metric learning loss
trainer_arg.add_argument('--neg_thresh', type=float, default=1.4)
trainer_arg.add_argument('--pos_thresh', type=float, default=0.1)
trainer_arg.add_argument('--alpha', type=float, default=0.01)
trainer_arg.add_argument('--beta', type=float, default=3)
trainer_arg.add_argument('--rte_thresh', type=int, default=30)
trainer_arg.add_argument('--rre_thresh', type=int, default=15)
trainer_arg.add_argument('--overlap_radius', type=float, default=0.0375)

# Data augmentation
trainer_arg.add_argument('--use_random_scale', type=str2bool, default=False)
trainer_arg.add_argument('--min_scale', type=float, default=0.8)
trainer_arg.add_argument('--max_scale', type=float, default=1.2)
trainer_arg.add_argument('--use_random_rotation', type=str2bool, default=False)
trainer_arg.add_argument('--rotation_range', type=float, default=360)


# Data loader configs
trainer_arg.add_argument('--train_phase', type=str, default="train")
trainer_arg.add_argument('--val_phase', type=str, default="val")
trainer_arg.add_argument('--test_phase', type=str, default="test")
trainer_arg.add_argument('--test_valid', type=str2bool, default=True)
trainer_arg.add_argument(
    '--positive_pair_search_voxel_size_multiplier', type=float, default=1.5)


# Network specific configurations

network_arg = add_argument_group('Network')
network_arg.add_argument('--model', type=str, default='ResUNetBN2C')
network_arg.add_argument('--model_select', type=str, default='simple')

network_arg.add_argument('--sparse_dims', type=int, default=32, help='Feature dimension')
network_arg.add_argument('--conv1_kernel_size', type=int, default=5)
network_arg.add_argument('--eps', type=float, default=1e-12)
network_arg.add_argument('--dropout', type=float, default=0.0, metavar='N',
                        help='Dropout ratio in transformer')

# Predator
network_arg.add_argument('--in_feats_dim', type=int, default=1)
network_arg.add_argument('--out_feats_dim', type=int, default=32)
network_arg.add_argument('--gnn_feats_dim', type=int, default=256)
network_arg.add_argument('--num_head', type=int, default=4, metavar='N',
                        help='Num of heads in multiheadedattention')
network_arg.add_argument('--dgcnn_k', type=int, default=10)
network_arg.add_argument('--nets', type=str, default=['self','cross','self'])
network_arg.add_argument('--dist_type', type=str, default='L2')
network_arg.add_argument('--normalize_feature', type=str2bool, default='True')
network_arg.add_argument('--best_val_metric', type=str, default='recall')

network_arg.add_argument('--num_trial', default=100000, type=int)
network_arg.add_argument('--r_binsize', default=0.015, type=float)
network_arg.add_argument('--t_binsize', default=0.015, type=float)

# Attention specific configurations
att_arg = add_argument_group('Attention')
att_arg.add_argument('--radius', type=float, default=0.1)
att_arg.add_argument('--kernel_size', type=int, default=16)

# Optimizer arguments
opt_arg = add_argument_group('Optimizer')
opt_arg.add_argument('--optimizer', type=str, default='Adam')
opt_arg.add_argument('--max_epoch', type=int, default=10)
opt_arg.add_argument('--lr', type=float, default=1e-2)
opt_arg.add_argument('--momentum', type=float, default=0.9)
opt_arg.add_argument('--sgd_momentum', type=float, default=0.9)
opt_arg.add_argument('--sgd_dampening', type=float, default=0.1)
opt_arg.add_argument('--adam_beta1', type=float, default=0.9)
opt_arg.add_argument('--adam_beta2', type=float, default=0.999)
opt_arg.add_argument('--weight_decay', type=float, default=1e-4)
opt_arg.add_argument('--iter_size', type=int, default=2, help='accumulate gradient')
opt_arg.add_argument('--bn_momentum', type=float, default=0.05)
opt_arg.add_argument('--exp_gamma', type=float, default=0.99)
opt_arg.add_argument('--scheduler', type=str, default='ExpLR')
opt_arg.add_argument(
    '--icp_cache_path', type=str, default="../datasets/FCGF/kitti/icp/")

misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--use_gpu', type=str2bool, default=True)

misc_arg.add_argument('--weights_dir', type=str, default=None)
misc_arg.add_argument('--resume', type=str, default=None)
misc_arg.add_argument('--resume_dir', type=str, default=None)
misc_arg.add_argument('--train_num_thread', type=int, default=2)
misc_arg.add_argument('--val_num_thread', type=int, default=1)
misc_arg.add_argument('--test_num_thread', type=int, default=1)
misc_arg.add_argument('--fast_validation', type=str2bool, default=False)
misc_arg.add_argument(
    '--nn_max_n',
    type=int,
    default=500,
    help='The maximum number of features to find nearest neighbors in batch')

# BackBone Network
misc_arg.add_argument('--feat_weight', type=str, default='./weights/fcgf_3dmatch_25.pth')


data_arg = add_argument_group('Data')
data_arg.add_argument('--dataset', type=str, default='ThreeDMatchPairDataset')
data_arg.add_argument('--voxel_size', type=float, default=0.025)
data_arg.add_argument(
    '--threed_match_dir', type=str, default="../Datasets/3dmatch/threedmatch")

def get_config():
  args = parser.parse_args()
  return args

