import argparse
import os
import torch


def get_args():
    parser = argparse.ArgumentParser(description='Learning Visual Embeddings')
    parser.add_argument('--project-dir', metavar='PROJECT_DIR', dest='project_dir', \
                        help='path to project directory', required=False, default='.')
    parser.add_argument('--data-dir', metavar='DATA_DIR', dest='data_dir', \
                        help='path to data directory (used if different from "data")', \
                        required=False, default='data')
    parser.add_argument('--plots-dir', metavar='PLOTS_DIR', dest='plots_dir', \
                        help='path to plots directory', required=False)
    parser.add_argument('--logs-dir', metavar='LOGS_DIR', dest='logs_dir', \
                        help='path to logs directory', required=False)
    parser.add_argument('--checkpoints-dir', metavar='CHECKPOINTS_DIR', dest='checkpoints_dir', \
                        help='path to checkpoints directory', required=False)
    parser.add_argument('--embeddings-dir', metavar='EMBEDDINGS_DIR', dest='embeddings_dir', \
                        help='path to embeddings directory', required=False)
    parser.add_argument('--dataset-type', metavar='DATASET_TYPE', dest='dataset_type', \
                        help='name of PyTorch Dataset to use: maze | fixed_mmnist | random_mmnist', \
                        required=False, default='maze')
    parser.add_argument('--dataset', metavar='DATASET', dest='dataset_name', \
                        help='name of dataset file in data directory', \
                        required=False, default='all_mazes_16_3_6')
    parser.add_argument('--data-ext', metavar='DATA_EXT', dest='data_ext', \
                        help='extension of dataset file in data directory', \
                        required=False, default='.npy')
    parser.add_argument('--offline', action='store_true', \
                        help='use offline preprocessing of data loader')
    parser.add_argument('--force', action='store_true', \
                        help='overwrites all existing dumped data sets (if used with `--offline`)')
    parser.add_argument('--cpu', action='store_true', help='use CPU')
    parser.add_argument('--cuda', action='store_true', help='use CUDA, default id: 0')
    parser.add_argument('--device', metavar='DEVICE', dest='device', \
                        help='device', default='cuda', required=False)
    parser.add_argument('--device-ids', metavar='DEVICE_IDS', dest='device_ids', help='IDs of GPUs to use', \
                        required=False, type=eval, default='[0]')
    parser.add_argument('--parallel', action='store_true', help='use all GPUs available', required=False)
    parser.add_argument('--emb-model', metavar='EMB_MODEL', dest='emb_model', \
                        help='name of embedding network', required=False, default='emb-cnn1')
    parser.add_argument('--load-ckpt', metavar='LOAD_CHECKPOINT', dest='load_ckpt', \
                        help='name of checkpoint file to load', required=False)
    parser.add_argument('--load-emb-ckpt', metavar='LOAD_EMB_CKPT', dest='load_emb_ckpt', \
                        help='name of embedding network file to load', required=False)
    parser.add_argument('--load-cls-ckpt', metavar='LOAD_CLS_CKPT', dest='load_cls_ckpt', \
                        help='name of classification network file to load', required=False)
    parser.add_argument('--batch-size', metavar='BATCH_SIZE', dest='batch_size', help='batch size', \
                        required=False, type=int, default=64)
    parser.add_argument('--epochs', metavar='EPOCHS', dest='epochs', help='number of epochs', \
                        required=False, type=int, default=10)
    parser.add_argument('--lr', metavar='LR', dest='lr', help='learning rate', \
                        required=False, type=float, default=1e-4)
    parser.add_argument('--flatten', action='store_true', help='flatten data into 1 long video')
    parser.add_argument('--num-train', metavar='NUM_TRAIN', dest='num_train', \
                        help='number of training examples', required=False, type=int, default=50000)
    parser.add_argument('--num-frames', metavar='NUM_FRAMES_IN_STACK', dest='num_frames', \
                        help='number of stacked frames', required=False, type=int, default=2)
    parser.add_argument('--num-channels', metavar='NUM_CHANNELS', dest='num_channels', \
                        help='number of channels in input image', required=False, type=int, default=1)
    parser.add_argument('--num-pairs', metavar='NUM_PAIRS_PER_EXAMPLE', dest='num_pairs', \
                        help='number of pairs per video', required=False, type=int, default=5)
    parser.add_argument('--use-pool', action='store_true', \
                        help='use pooling instead of strided convolutions')
    parser.add_argument('--use-res', action='store_true', help='use residual layers')

    args = parser.parse_args()

    args.logs_dir = args.logs_dir if args.logs_dir else os.path.join(args.project_dir, 'logs')
    args.plots_dir = args.plots_dir if args.plots_dir else os.path.join(args.project_dir, 'plots')
    args.checkpoints_dir = args.checkpoints_dir if args.checkpoints_dir \
                                                else os.path.join(args.project_dir, 'checkpoints')
    args.embeddings_dir = args.embeddings_dir if args.embeddings_dir \
                                              else os.path.join(args.checkpoints_dir, 'embeddings')
    args.time_buckets = [[0], [1], [2], [3,4], range(5,11,1)]

    args.test_size, args.val_size = 0.2, 0.2
    if not args.offline:
        args.train_size = 1 - args.test_size - args.val_size
        args.num_test = int((args.test_size/args.train_size)*args.num_train)
        args.num_val = int((args.val_size/args.train_size)*args.num_train)

    if (args.device == 'cuda' or args.cuda) and torch.cuda.is_available():
        total_gpus = torch.cuda.device_count() # Total number of GPUs available
        if args.cuda: # Train on 1 GPU
            args.device_ids = [0]
        elif args.parallel: # Train on all GPUs
            args.device_ids = range(total_gpus)
        else: # Train on specified GPUs
            assert total_gpus >= len(args.device_ids), '{} GPUs not available! Only {} GPU(s) available'.format(len(args.device_ids), total_gpus)
    else:
        args.device = 'cpu'

    return args
