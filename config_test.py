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

logging_arg.add_argument('--snapshot_dir', type=str, default='outputs/snapshot')

trainer_arg = add_argument_group('Trainer')

trainer_arg.add_argument('--test_phase', type=str, default="test")
trainer_arg.add_argument('--val_batch_size', type=int, default=1)
trainer_arg.add_argument(
    '--positive_pair_search_voxel_size_multiplier', type=float, default=1.5)
trainer_arg.add_argument('--min_scale', type=float, default=0.8)
trainer_arg.add_argument('--max_scale', type=float, default=1.2)
trainer_arg.add_argument('--use_random_rotation', type=str2bool, default=False)
trainer_arg.add_argument('--rotation_range', type=float, default=360)
trainer_arg.add_argument('--overlap_radius', type=float, default=0.45)
trainer_arg.add_argument('--num_points', type=int, default=4096)
trainer_arg.add_argument('--sample_num_points', type=int, default=512)
trainer_arg.add_argument('--max_points', type=int, default=1024)



# dNetwork specific configurations
dgf_arg = add_argument_group('Network')
dgf_arg.add_argument('--sparse_dims', type=int, default=32, help='Feature dimension')
dgf_arg.add_argument('--emb_dims', type=int, default=32, metavar='N',help='Dimension of embeddings')
dgf_arg.add_argument('--dropout', type=float, default=0.0, metavar='N',
                        help='Dropout ratio in transformer')
dgf_arg.add_argument('--n_heads', type=int, default=4, metavar='N',
                        help='Num of heads in multiheadedattention')
dgf_arg.add_argument('--ff_dims', type=int, default=32, metavar='N',
                        help='Num of dimensions of fc in transformer')
dgf_arg.add_argument('--conv1_kernel_size', type=int, default=5)
dgf_arg.add_argument('--conv2_kernel_size', type=int, default=5)
dgf_arg.add_argument('--alpha_factor', type=float, default=4)
dgf_arg.add_argument('--eps', type=float, default=1e-12)
dgf_arg.add_argument('--dist_type', type=str, default='L2')
dgf_arg.add_argument('--best_val_metric1', type=str, default='loss')
dgf_arg.add_argument('--best_val_metric2', type=str, default='recall')
dgf_arg.add_argument('--k_nn_geof', default=32, type=int, help='number of neighbors to describe the local geometry')

dgf_arg.add_argument('--num_trial', default=100000, type=int)
dgf_arg.add_argument('--r_binsize', default=0.005, type=float)
dgf_arg.add_argument('--t_binsize', default=0.05, type=float)

# Attention specific configurations
att_arg = add_argument_group('Attention')
att_arg.add_argument('--radius', type=float, default=0.1)
att_arg.add_argument('--kernel_size', type=int, default=16)

# Optimizer arguments
opt_arg = add_argument_group('Optimizer')
opt_arg.add_argument('--optimizer', type=str, default='SGD')
opt_arg.add_argument('--max_epoch', type=int, default=10)
opt_arg.add_argument('--lr', type=float, default=1e-1)
opt_arg.add_argument('--momentum', type=float, default=0.7)
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
misc_arg.add_argument('--verbose', type=str2bool, default=True)
misc_arg.add_argument('--verbose_freq', type=int, default=100)
misc_arg.add_argument('--weights', type=str, default=None)
misc_arg.add_argument('--feat_weight', type=str, default='./weights/fcgf_kitti_30.pth')
misc_arg.add_argument('--resume', type=str, default=None)
misc_arg.add_argument('--resume_dir', type=str, default='./outputs')
misc_arg.add_argument('--train_num_thread', type=int, default=2)
misc_arg.add_argument('--val_num_thread', type=int, default=1)
misc_arg.add_argument('--test_num_thread', type=int, default=1)
misc_arg.add_argument('--fast_validation', type=str2bool, default=False)
misc_arg.add_argument(
    '--nn_max_n',
    type=int,
    default=500,
    help='The maximum number of features to find nearest neighbors in batch')

data_arg = add_argument_group('Data')
data_arg.add_argument('--data_augmentation', type=str2bool, default=False)

data_arg.add_argument('--augment_noise', type=float, default=0.01)
data_arg.add_argument('--augment_shift_range', type=float, default=2.0)
data_arg.add_argument('--augment_scale_max', type=float, default=1.2)
data_arg.add_argument('--augment_scale_min', type=float, default=0.8)
data_arg.add_argument('--dataset', type=str, default='KITTIDataset')
data_arg.add_argument("--run_name",default="test", type=str, required=False, help="experiment title")

data_arg.add_argument('--voxel_size', type=float, default=0.3)
data_arg.add_argument(
    '--threed_match_dir', type=str, default="../Datasets/3dmatch/threedmatch_test")
    #'--threed_match_dir', type=str, default="./data/indoor")
data_arg.add_argument(
        '--source', default='../Datasets/kitti', type=str, help='path to 3dmatch test dataset')
data_arg.add_argument(
    '--target', default='./features_tmp/', type=str, help='path to produce generated data')
data_arg.add_argument(
    '--model', default='./outputs/best_val_recall_checkpoint.pth', type=str,  help='path to checkpoint')

data_arg.add_argument(
    '--out_dir',
    type=str,
    default='./experiments',
    help='path to save benchmark results',
)
data_arg.add_argument(
    '--num_rand_keypoints',
    type=int,
    default=5000,
    help='Number of random keypoints for each scene')
data_arg.add_argument(
    '--kitti_root', type=str, default="../Datasets/kitti")
data_arg.add_argument(
    '--kitti_max_time_diff',
    type=int,
    default=3,
    help='max time difference between pairs (non inclusive)')
data_arg.add_argument('--kitti_date', type=str, default='2011_09_26')

def get_config():
  args = parser.parse_args()
  return args

