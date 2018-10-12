import numpy as np
import pandas as pd
import argparse
import os
import torch
from torchvision.utils import make_grid

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--project-dir', metavar='PROJECT_DIR', dest='project_dir', help='path to project directory', required=False)
parser.add_argument('--dataset', metavar='DATASET', dest='dataset', help='name of dataset file in data directory', required=False)
parser.add_argument('--data-dir', metavar='DATA_DIR', dest='data_dir', help='path to data directory (used if different from "data")', \
                    required=False, default='data')
parser.add_argument('--img-dim', metavar='IMG_DIM', dest='img_dim', help='height (or width) of output video', required=False, \
                    type=int, default=64)
parser.add_argument('--seq-len', metavar='SEQ_LEN', dest='seq_len', help='sequence length of output video', required=False, \
                    type=int, default=20)
args = parser.parse_args()


PROJECT_DIR = args.project_dir if args.project_dir else '/home/mihir/Desktop/GitHub/nyu/learning_visual_embeddings/'
DATA_DIR, PLOTS_DIR = args.data_dir, 'plots'
SEQ_LEN = args.seq_len
IMG_DIM = args.img_dim

one_image = np.zeros(shape=(IMG_DIM, IMG_DIM))
velocities = [1, 3]

data = np.array([])
for c in range(len(velocities)):
    for s_idx in range(one_image.shape[0]):
        temp_seq = np.array([])
        i = s_idx
        for seq_num in range(SEQ_LEN):
            temp = np.copy(one_image)
            temp[:,i] = np.ones(temp.shape[0])
            temp_seq = np.append(temp_seq, temp)
            i = (i + velocities[c]) % one_image.shape[0]
        temp_seq = np.reshape(temp_seq,(1, -1, IMG_DIM, IMG_DIM))
        data = temp_seq if not data.size else np.vstack((data, temp_seq))

data = np.reshape(data, (SEQ_LEN, -1, IMG_DIM, IMG_DIM))
np.save(open(os.path.join(DATA_DIR, 'moving_bars_{}_{}.npy'.format(SEQ_LEN, IMG_DIM)), 'wb'), data)
