import numpy as np
import pandas as pd
import os
import argparse
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from capstone_project.preprocessing import generate_all_offline_dataloaders, generate_online_dataloader
from capstone_project.models.embedding_network import EmbeddingNetwork
from capstone_project.models.classification_network import ClassificationNetwork
from capstone_project.utils import *


parser = argparse.ArgumentParser()
parser.add_argument('--project-dir', metavar='PROJECT_DIR', dest='project_dir', help='path to project directory', required=False, default='.')
parser.add_argument('--dataset', metavar='DATASET', dest='dataset', help='name of dataset file in data directory', required=False, \
					default='mnist_test_seq')
parser.add_argument('--data-ext', metavar='DATA_EXT', dest='data_ext', help='extension of dataset file in data directory', required=False, \
					default='.npy')
parser.add_argument('--data-dir', metavar='DATA_DIR', dest='data_dir', help='path to data directory (used if different from "data")', \
					required=False, default='data')
parser.add_argument('--offline', action='store_true', help='use offline preprocessing of data loader')
parser.add_argument('--checkpoints-dir', metavar='CHECKPOINTS_DIR', dest='checkpoints_dir', help='path to checkpoints directory', \
					required=False, default='checkpoints')
parser.add_argument('--load-ckpt', metavar='LOAD_CHECKPOINT', dest='load_ckpt', help='name of checkpoint file to load', required=False)
parser.add_argument('--batch-size', metavar='BATCH_SIZE', dest='batch_size', help='batch size', required=False, type=int, default=64)
parser.add_argument('--epochs', metavar='EPOCHS', dest='epochs', help='number of epochs', required=False, type=int, default=10)
parser.add_argument('--device', metavar='DEVICE', dest='device', help='device', required=False)
parser.add_argument('--device-id', metavar='DEVICE_ID', dest='device_id', help='device id of gpu', required=False, type=int)
parser.add_argument('--ngpu', metavar='NGPU', dest='ngpu', help='number of GPUs to use (0,1,...,ngpu-1)', required=False, type=int)
parser.add_argument('--parallel', action='store_true', help='use all GPUs available', required=False)
parser.add_argument('--lr', metavar='LR', dest='lr', help='learning rate', required=False, type=float, default=1e-4)
parser.add_argument('--num-train', metavar='NUM_TRAIN', dest='num_train', help='number of training examples', required=False, \
					type=int, default=50000)
parser.add_argument('--num-frames', metavar='NUM_FRAMES_IN_STACK', dest='num_frames', help='number of stacked frames', required=False, \
					type=int, default=2)
parser.add_argument('--num-pairs', metavar='NUM_PAIRS_PER_EXAMPLE', dest='num_pairs', help='number of pairs per video', required=False, \
					type=int, default=5)
parser.add_argument('--use-pool', action='store_true', help='use pooling instead of strided convolutions')
parser.add_argument('--use-res', action='store_true', help='use residual layers')
parser.add_argument('--force', action='store_true', help='overwrites all existing dumped data sets (if used with `--offline`)')
args = parser.parse_args()


# Globals
PROJECT_DIR = args.project_dir
DATA_DIR,  PLOTS_DIR, LOGGING_DIR = args.data_dir, 'plots', 'logs'
CHECKPOINTS_DIR, CHECKPOINT_FILE = args.checkpoints_dir, args.load_ckpt
DATASET, DATA_EXT = args.dataset, args.data_ext
OFFLINE = args.offline

TEST_SIZE, VAL_SIZE = 0.2, 0.2
if not OFFLINE:
	NUM_TRAIN = args.num_train
	TRAIN_SIZE = 1 - TEST_SIZE - VAL_SIZE
	NUM_TEST, NUM_VAL = int((TEST_SIZE/TRAIN_SIZE)*NUM_TRAIN), int((VAL_SIZE/TRAIN_SIZE)*NUM_TRAIN)

BATCH_SIZE = args.batch_size    # input batch size for training
N_EPOCHS = args.epochs          # number of epochs to train
LR = args.lr                    # learning rate
NGPU = args.ngpu                # number of GPUs
PARALLEL = args.parallel 		# use all GPUs

TOTAL_GPUs = torch.cuda.device_count() # Number of total GPUs available

if NGPU:
	assert TOTAL_GPUs >= NGPU, '{} GPUs not available! Only {} GPU(s) available'.format(NGPU, TOTAL_GPUs)

DEVICE = args.device if args.device else 'cuda' if torch.cuda.is_available() else 'cpu'
if args.device_id and 'cuda' in DEVICE:
	DEVICE_ID = args.device_id
	torch.cuda.set_device(DEVICE_ID)

NUM_FRAMES_IN_STACK = args.num_frames		# number of (total) frames to concatenate for each video
NUM_PAIRS_PER_EXAMPLE = args.num_pairs      # number of pairs to generate for given video and time difference
USE_POOL = args.use_pool					# use pooling instead of strided convolutions
USE_RES = args.use_res						# use residual layers
TIME_BUCKETS = [[0], [1], [2], [3,4], range(5,11,1)]

def main():
	torch.set_num_threads(1) # Prevent error on KeyboardInterrupt with multiple GPUs

	make_dirs(PROJECT_DIR, [CHECKPOINTS_DIR, PLOTS_DIR, LOGGING_DIR]) # Create all required directories if not present
	setup_logging(PROJECT_DIR, LOGGING_DIR) # Setup configuration for logging

	global_vars = globals().copy()
	print_config(global_vars) # Print all global variables defined above

	if OFFLINE:
		train_loader, val_loader, test_loader = generate_all_offline_dataloaders(PROJECT_DIR, DATA_DIR, PLOTS_DIR, DATASET, \
																				TIME_BUCKETS, BATCH_SIZE, NUM_PAIRS_PER_EXAMPLE, \
																				NUM_FRAMES_IN_STACK, DATA_EXT, args.force)
	else:
		train_loader, transforms, data_max, data_min = generate_online_dataloader(PROJECT_DIR, DATA_DIR, PLOTS_DIR, DATASET, \
																				NUM_TRAIN, 'train', TIME_BUCKETS, BATCH_SIZE, \
																				NUM_FRAMES_IN_STACK, DATA_EXT)
		val_loader = generate_online_dataloader(PROJECT_DIR, DATA_DIR, PLOTS_DIR, DATASET, NUM_VAL, 'val', \
										TIME_BUCKETS, BATCH_SIZE, NUM_FRAMES_IN_STACK, DATA_EXT, transforms, data_max, data_min)
		test_loader = generate_online_dataloader(PROJECT_DIR, DATA_DIR, PLOTS_DIR, DATASET, NUM_TEST, 'test', \
										TIME_BUCKETS, BATCH_SIZE, NUM_FRAMES_IN_STACK, DATA_EXT, transforms, data_max, data_min)

	# Network hyperparameters
	img_dim = train_loader.dataset.__getitem__(0)[0].shape[-1]
	in_dim, in_channels, out_dim = img_dim, NUM_FRAMES_IN_STACK, 1024
	embedding_hidden_size, classification_hidden_size = 1024, 1024
	num_outputs = len(TIME_BUCKETS)

	start_epoch = 0 # Initialize starting epoch number (used later if checkpoint loaded)
	stop_epoch = N_EPOCHS+start_epoch # Store epoch upto which model is trained (used in case of KeyboardInterrupt)

	logging.info('Creating models...')
	embedding_network = EmbeddingNetwork(in_dim, in_channels, embedding_hidden_size, out_dim, use_pool=USE_POOL, use_res=USE_RES)
	classification_network = ClassificationNetwork(out_dim, classification_hidden_size, num_outputs)
	logging.info('Done.')

	# Define criteria and optimizer
	criterion_train = nn.CrossEntropyLoss()
	criterion_test = nn.CrossEntropyLoss(reduction='sum')
	optimizer = optim.Adam(list(embedding_network.parameters()) + list(classification_network.parameters()), lr=LR)

	train_loss_history, train_accuracy_history = [], []
	val_loss_history, val_accuracy_history = [], []

	# Load model state dicts if required
	if CHECKPOINT_FILE:
		embedding_network, classification_network, optimizer, train_loss_history, val_loss_history, \
		train_accuracy_history, val_accuracy_history, epoch_trained = \
			load_checkpoint(embedding_network, classification_network, optimizer, CHECKPOINT_FILE, PROJECT_DIR, CHECKPOINTS_DIR, DEVICE)
		start_epoch = epoch_trained # Start from (epoch_trained+1) if checkpoint loaded

	# Check if model is to be parallelized
	if TOTAL_GPUs > 1 and (NGPU or PARALLEL):
		DEVICE_IDs = range(TOTAL_GPUs) if PARALLEL else range(NGPU)
		logging.info('Using {} GPUs...'.format(len(DEVICE_IDs)))
		embedding_network = nn.DataParallel(embedding_network, device_ids=DEVICE_IDs)
		classification_network = nn.DataParallel(classification_network, device_ids=DEVICE_IDs)
		logging.info('Done.')
	embedding_network = embedding_network.to(DEVICE)
	classification_network = classification_network.to(DEVICE)

	for epoch in range(start_epoch+1, N_EPOCHS+start_epoch+1):
		try:
			train_losses = train(
				embedding_network=embedding_network,
				classification_network=classification_network,
				criterion=criterion_train,
				dataloader=train_loader,
				optimizer=optimizer,
				device=DEVICE,
				epoch=epoch,
				offline=OFFLINE
			)

			val_loss, val_pred, val_true = test(
				embedding_network=embedding_network,
				classification_network=classification_network,
				dataloader=val_loader,
				criterion=criterion_test,
				device=DEVICE,
				offline=OFFLINE
			)

			accuracy_train = accuracy(embedding_network, classification_network, train_loader, criterion_test, DEVICE)
			accuracy_val = accuracy(embedding_network, classification_network, val_loader, criterion_test, DEVICE)
			train_loss_history.extend(train_losses)
			val_loss_history.append(val_loss)
			train_accuracy_history.append(accuracy_train)
			val_accuracy_history.append(accuracy_val)

			logging.info('TRAIN Epoch: {}\tAverage loss: {:.4f}, Accuracy: {:.0f}%'.format(epoch, np.sum(train_losses), accuracy_train))
			logging.info('VAL   Epoch: {}\tAverage loss: {:.4f}, Accuracy: {:.0f}%\n'.format(epoch, val_loss, accuracy_val))
		except KeyboardInterrupt:
			logging.info('Keyboard Interrupted!')
			stop_epoch = epoch-1
			break

	# Save the model checkpoints
	logging.info('Dumping model and results...')
	print_config(global_vars) # Print all global variables before saving checkpointing
	save_checkpoint(embedding_network, classification_network, optimizer, train_loss_history, val_loss_history, \
					train_accuracy_history, val_accuracy_history, stop_epoch, DATASET, NUM_FRAMES_IN_STACK, \
					NUM_PAIRS_PER_EXAMPLE, PROJECT_DIR, CHECKPOINTS_DIR, PARALLEL or NGPU)
	logging.info('Done.')

	if len(train_loss_history):
		logging.info('Saving and plotting loss and accuracy histories...')
		fig = plt.figure(figsize=(10,8))
		plt.plot(train_loss_history, alpha=0.5, color='blue', label='train')
		xticks = [epoch*len(train_loader) for epoch in range(1, len(val_loss_history)+1)]
		plt.plot(xticks, val_loss_history, alpha=0.5, color='orange', label='test')
		plt.legend()
		plt.title('Loss vs. Iterations')
		save_plot(PROJECT_DIR, PLOTS_DIR, fig, 'loss_vs_iterations.png')

		fig = plt.figure(figsize=(10,8))
		plt.plot(train_accuracy_history, alpha=0.5, color='blue', label='train')
		plt.plot(val_accuracy_history, alpha=0.5, color='orange', label='test')
		plt.legend()
		plt.title('Accuracy vs. Iterations')
		save_plot(PROJECT_DIR, PLOTS_DIR, fig, 'accuracies_vs_iterations.png')
		logging.info('Done.')

if __name__ == '__main__':
	main()
