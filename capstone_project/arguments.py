import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Learning Visual Embeddings')
    parser.add_argument('--project-dir', metavar='PROJECT_DIR', dest='project_dir', \
                        help='path to project directory', required=False, default='.')
    parser.add_argument('--dataset-type', metavar='DATASET_TYPE', dest='dataset_type', \
                        help='name of PyTorch Dataset to use: maze | fixed_mmnist | random_mmnist', \
                        required=False, default='maze')
    parser.add_argument('--dataset', metavar='DATASET', dest='dataset', \
                        help='name of dataset file in data directory', \
                        required=False, default='mnist_test_seq')
    parser.add_argument('--data-ext', metavar='DATA_EXT', dest='data_ext', \
                        help='extension of dataset file in data directory', \
                        required=False, default='.npy')
    parser.add_argument('--data-dir', metavar='DATA_DIR', dest='data_dir', \
                        help='path to data directory (used if different from "data")', \
                        required=False, default='data')
    parser.add_argument('--offline', action='store_true', \
                        help='use offline preprocessing of data loader')
    parser.add_argument('--emb-model', metavar='EMB_MODEL', dest='emb_model', \
                        help='name of embedding network', required=False, default='cnn')
    parser.add_argument('--checkpoints-dir', metavar='CHECKPOINTS_DIR', dest='checkpoints_dir', \
                        help='path to checkpoints directory', required=False, default='checkpoints')
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
    parser.add_argument('--device', metavar='DEVICE', dest='device', \
                        help='device', required=False)
    parser.add_argument('--device-id', metavar='DEVICE_ID', dest='device_id', \
                        help='device id of gpu', required=False, type=int)
    parser.add_argument('--ngpu', metavar='NGPU', dest='ngpu', \
                        help='number of GPUs to use (0,1,...,ngpu-1)', required=False, type=int)
    parser.add_argument('--parallel', action='store_true', help='use all GPUs available', required=False)
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
    parser.add_argument('--force', action='store_true', \
                        help='overwrites all existing dumped data sets (if used with `--offline`)')
    args = parser.parse_args()

    return args
