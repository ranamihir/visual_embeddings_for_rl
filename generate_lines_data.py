import numpy as np
import pandas as pd
import argparse
import os
import torch
from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser()
parser.add_argument('--project-dir', metavar='PROJECT_DIR', dest='project_dir', help='path to project directory', required=False)
parser.add_argument('--dataset', metavar='DATASET', dest='dataset', help='name of dataset file in data directory', required=False)
parser.add_argument('--data-dir', metavar='DATA_DIR', dest='data_dir', help='path to data directory (used if different from "data")', \
					required=False, default='data')
parser.add_argument('--img-dim', metavar='IMG_DIM', dest='img_dim', help='height (or width) of output video', required=False, \
					type=int, default=121)
parser.add_argument('--seq-len', metavar='SEQ_LEN', dest='seq_len', help='sequence length of output video', required=False, \
					type=int, default=20)
args = parser.parse_args()


# Globals
PROJECT_DIR = args.project_dir if args.project_dir else '/home/mihir/Desktop/GitHub/nyu/learning_visual_embeddings/'
DATA_DIR = args.data_dir
SEQ_LEN = args.seq_len
IMG_DIM = args.img_dim
VELOCITIES = [1, 3]
TEST_SIZE, VAL_SIZE = 0.2, 0.2

def generate_data(img_dim, seq_len, velocities):
	data = np.array([])
	image = np.zeros((img_dim, img_dim))

	for velocity in velocities:
		for s_idx in range(img_dim):
			temp_seq = np.array([])
			i = s_idx
			for seq_num in range(seq_len):
				temp = np.copy(image)
				temp[:, i] = np.ones(img_dim)
				temp_seq = np.append(temp_seq, temp)
				i = (i + velocity) % img_dim
			temp_seq = np.reshape(temp_seq, (1, -1, img_dim, img_dim))
			data = np.vstack((data, temp_seq)) if data.size else temp_seq

	return data

def split_and_dump_data(data, val_size, test_size, project_dir, data_dir):
	num_test = int(np.floor(test_size*len(data)))
	num_train_val = len(data) - num_test
	num_val = int(np.floor(num_train_val*val_size/(1 - test_size)))

	train_data, test_data = train_test_split(data, test_size=num_test, random_state=1337)
	train_data, val_data = train_test_split(train_data, test_size=num_val, random_state=1337)

	data_dict = {
		'train': train_data,
		'val': val_data,
		'test': test_data
	}

	np.save(open(os.path.join(DATA_DIR, 'moving_bars_{}_{}.npy'.format(SEQ_LEN, IMG_DIM)), 'wb'), data.swapaxes(0,1))
	for key in data_dict:
		np.save(open(os.path.join(DATA_DIR, 'moving_bars_{}_{}_{}.npy'.format(SEQ_LEN, IMG_DIM, key)), 'wb'), data_dict[key].swapaxes(0,1))

def main():
	data = generate_data(IMG_DIM, SEQ_LEN, VELOCITIES)
	split_and_dump_data(data, VAL_SIZE, TEST_SIZE, PROJECT_DIR, DATA_DIR)

if __name__ == '__main__':
	main()
