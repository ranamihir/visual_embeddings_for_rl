import numpy as np
import pandas as pd
import logging

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from capstone_project.arguments import get_args
from capstone_project.preprocessing import generate_all_offline_dataloaders, generate_online_dataloader
from capstone_project.models.embedding_network import EmbeddingNetwork
from capstone_project.models.classification_network import ClassificationNetwork
from capstone_project.utils import *


args = get_args()


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
		train_loader, transforms = generate_online_dataloader(PROJECT_DIR, DATA_DIR, PLOTS_DIR, DATASET, \
															NUM_TRAIN, 'train', TIME_BUCKETS, BATCH_SIZE, \
															NUM_FRAMES_IN_STACK, DATA_EXT)
		val_loader = generate_online_dataloader(PROJECT_DIR, DATA_DIR, PLOTS_DIR, DATASET, NUM_VAL, 'val', \
										TIME_BUCKETS, BATCH_SIZE, NUM_FRAMES_IN_STACK, DATA_EXT, transforms)
		test_loader = generate_online_dataloader(PROJECT_DIR, DATA_DIR, PLOTS_DIR, DATASET, NUM_TEST, 'test', \
										TIME_BUCKETS, BATCH_SIZE, NUM_FRAMES_IN_STACK, DATA_EXT, transforms)

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
	if TOTAL_GPUs > 1 and (PARALLEL or NGPU):
		DEVICE_IDs = range(TOTAL_GPUs) if PARALLEL else range(NGPU)
		logging.info('Using {} GPUs...'.format(len(DEVICE_IDs)))
		embedding_network = nn.DataParallel(embedding_network, device_ids=DEVICE_IDs)
		classification_network = nn.DataParallel(classification_network, device_ids=DEVICE_IDs)
		logging.info('Done.')
	embedding_network = embedding_network.to(DEVICE)
	classification_network = classification_network.to(DEVICE)

	early_stopping = EarlyStopping(mode='minimize', min_delta=0, patience=10)
	best_epoch = start_epoch+1

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

			if early_stopping.is_better(val_loss):
				logging.info('Saving current best model checkpoint...')
				save_checkpoint(embedding_network, classification_network, optimizer, train_loss_history, val_loss_history, \
							train_accuracy_history, val_accuracy_history, epoch, DATASET, NUM_FRAMES_IN_STACK, \
							NUM_PAIRS_PER_EXAMPLE, PROJECT_DIR, CHECKPOINTS_DIR, USE_POOL, USE_RES, PARALLEL or NGPU)
				logging.info('Done.')
				logging.info('Removing previous best model checkpoint...')
				remove_checkpoint(DATASET, NUM_FRAMES_IN_STACK, NUM_PAIRS_PER_EXAMPLE, PROJECT_DIR, \
								CHECKPOINTS_DIR, best_epoch, USE_POOL, USE_RES)
				logging.info('Done.')
				best_epoch = epoch

			if early_stopping.stop(val_loss) or round(accuracy_val) == 100:
				logging.info('Stopping early after {} epochs.'.format(epoch))
				stop_epoch = epoch
				break
		except KeyboardInterrupt:
			logging.info('Keyboard Interrupted!')
			stop_epoch = epoch-1
			break

	# Save the model checkpoints
	logging.info('Dumping model and results...')
	print_config(global_vars) # Print all global variables before saving checkpointing
	save_checkpoint(embedding_network, classification_network, optimizer, train_loss_history, val_loss_history, \
					train_accuracy_history, val_accuracy_history, stop_epoch, DATASET, NUM_FRAMES_IN_STACK, \
					NUM_PAIRS_PER_EXAMPLE, PROJECT_DIR, CHECKPOINTS_DIR, USE_POOL, USE_RES, PARALLEL or NGPU)
	logging.info('Done.')

	if len(train_loss_history) and len(val_loss_history):
		logging.info('Plotting and saving loss histories...')
		fig = plt.figure(figsize=(10,8))
		plt.plot(train_loss_history, alpha=0.5, color='blue', label='train')
		xticks = [epoch*len(train_loader) for epoch in range(1, len(val_loss_history)+1)]
		plt.plot(xticks, val_loss_history, alpha=0.5, color='orange', label='test')
		plt.legend()
		plt.title('Loss vs. Iterations')
		save_plot(PROJECT_DIR, PLOTS_DIR, fig, 'loss_vs_iterations.png')
		logging.info('Done.')

	if len(train_accuracy_history) and len(val_accuracy_history):
		logging.info('Plotting and saving accuracy histories...')
		fig = plt.figure(figsize=(10,8))
		plt.plot(train_accuracy_history, alpha=0.5, color='blue', label='train')
		plt.plot(val_accuracy_history, alpha=0.5, color='orange', label='test')
		plt.legend()
		plt.title('Accuracy vs. Iterations')
		save_plot(PROJECT_DIR, PLOTS_DIR, fig, 'accuracies_vs_iterations.png')
		logging.info('Done.')

if __name__ == '__main__':
	main()
