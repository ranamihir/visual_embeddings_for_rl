import numpy as np
import pandas as pd
import os
import logging
import time
import sys
import pickle
from itertools import islice
from pprint import pformat

import torch
from torchvision.utils import make_grid

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def train(embedding_network, classification_network, dataloader, criterion, optimizer, device, epoch, offline=False):
	loss_hist = []
	dataloader_iterator = enumerate(dataloader) if offline else enumerate(islice(dataloader, len(dataloader)))
	for batch_idx, (x1, x2, y, differences, (frame_1, frame_2)) in dataloader_iterator:
		x1, x2, y = x1.to(device).float(), x2.to(device).float(), y.to(device).long()
		embedding_network.train()
		classification_network.train()

		optimizer.zero_grad()

		embedding_output1 = embedding_network(x1)
		embedding_output2 = embedding_network(x2)
		classification_output = classification_network(embedding_output1, embedding_output2)

		loss = criterion(classification_output, y)
		loss.backward()
		optimizer.step()

		# Accurately compute loss, because of different batch size
		loss_train = loss.item() * len(x1) / len(dataloader.dataset)
		loss_hist.append(loss_train)

		if (batch_idx+1) % (len(dataloader.dataset)//(5*y.shape[0])) == 0:
			logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch, (batch_idx+1) * y.shape[0], len(dataloader.dataset),
				100. * (batch_idx+1) / len(dataloader), loss.item()))

	optimizer.zero_grad()
	return loss_hist

def test(embedding_network, classification_network, dataloader, criterion, device, offline=False):
	embedding_network.eval()
	classification_network.eval()

	loss_test = 0.
	y_hist = []
	output_hist = []
	with torch.no_grad():
		dataloader_iterator = enumerate(dataloader) if offline else enumerate(islice(dataloader, len(dataloader)))
		for batch_idx, (x1, x2, y, differences, (frame_1, frame_2)) in dataloader_iterator:
			x1, x2, y = x1.to(device).float(), x2.to(device).float(), y.to(device).long()
			embedding_output1 = embedding_network(x1)
			embedding_output2 = embedding_network(x2)
			classification_output = classification_network(embedding_output1, embedding_output2)
			loss = criterion(classification_output, y)

			# Accurately compute loss, because of different batch size
			loss_test += loss.item() / len(dataloader.dataset)

			output_hist.append(classification_output)
			y_hist.append(y)

	return loss_test, torch.cat(output_hist, dim=0), torch.cat(y_hist, dim=0)

def accuracy(embedding_network, classification_network, dataloader, criterion, device):
	_, y_predicted, y_true = test(
		embedding_network=embedding_network,
		classification_network=classification_network,
		dataloader=dataloader,
		criterion=criterion,
		device=device
	)

	y_predicted = y_predicted.max(1)[1]
	return 100*y_predicted.eq(y_true.data.view_as(y_predicted)).float().mean().item()

def imshow(data, mean, std, project_dir, plots_dir, dataset):
	logging.info('Plotting sample data and saving to "{}_sample.png"...'.format(dataset))
	image_dim = data.shape[-1]
	images = data[np.random.RandomState(0).choice(len(data), size=1)]
	images = torch.from_numpy(images)

	images = make_grid(images[0].reshape(-1, 1, image_dim, image_dim), nrow=5, padding=5, pad_value=1)
	images = images*std + mean  # unnormalize
	np_image = images.numpy()

	fig = plt.figure(figsize=(30, 10))
	plt.imshow(np.transpose(np_image, axes=(1, 2, 0)))
	plt.tight_layout()
	save_plot(project_dir, plots_dir, fig, '{}_sample.png'.format(dataset))
	logging.info('Done.')

def plot_video(data, project_dir, plots_dir, dataset):
	file_name = '{}_video_sample.png'.format(dataset)
	logging.info('Plotting sample data and saving to "{}"...'.format(file_name))
	image_dim = data.shape[-1]
	images = torch.from_numpy(data)

	images = make_grid(images.reshape(-1, 1, image_dim, image_dim), nrow=10, padding=5, pad_value=1)
	np_image = images.numpy()

	fig = plt.figure(figsize=(30, 10))
	plt.imshow(np.transpose(np_image, axes=(1, 2, 0)))
	plt.tight_layout()
	save_plot(project_dir, plots_dir, fig, file_name)
	logging.info('Done.');

def print_config(vars_dict):
	vars_dict = {key: value for key, value in vars_dict.items() if key == key.upper()}
	logging.info(pformat(vars_dict))

def save_plot(project_dir, plots_dir, fig, filename):
	fig.savefig(os.path.join(project_dir, plots_dir, filename))

def make_dirs(parent_dir, directories_to_create):
	for directory in directories_to_create:
		directory_path = os.path.join(parent_dir, directory)
		if not os.path.exists(directory_path):
			os.makedirs(directory_path)

def setup_logging(project_dir, logging_dir):
	log_path = os.path.join(project_dir, logging_dir)
	filename = '{}.log'.format(time.strftime('%Y_%m_%d'))
	log_handlers = [logging.FileHandler(os.path.join(log_path, filename)), logging.StreamHandler()]
	logging.basicConfig(format='%(asctime)s: %(message)s', datefmt='%Y/%m/%d %I:%M:%S %p', \
						handlers=log_handlers, level=logging.DEBUG)
	logging.info('\n\n\n')

def save_object(object, filepath):
	'''
	This is a defensive way to write pickle.write, allowing for very large files on all platforms
	'''
	max_bytes = 2**31 - 1
	bytes_out = pickle.dumps(object, protocol=4)
	n_bytes = sys.getsizeof(bytes_out)
	with open(filepath, 'wb') as f_out:
		for idx in range(0, n_bytes, max_bytes):
			f_out.write(bytes_out[idx:idx+max_bytes])

def load_object(filepath):
	'''
	This is a defensive way to write pickle.load, allowing for very large files on all platforms
	'''
	max_bytes = 2**31 - 1
	try:
		input_size = os.path.getsize(filepath)
		bytes_in = bytearray(0)
		with open(filepath, 'rb') as f:
			for _ in range(0, input_size, max_bytes):
				bytes_in += f.read(max_bytes)
		object = pickle.loads(bytes_in)
	except:
		return None
	return object

def save_checkpoint(embedding_network, classification_network, optimizer, train_loss_history, val_loss_history, \
					train_accuracy_history, val_accuracy_history, epoch, dataset, num_frames_in_stack, \
					num_pairs_per_example, project_dir, checkpoints_dir, use_pool=False, use_res=False, is_parallel=False):

	state_dict = {
		'embedding_state_dict': embedding_network.module.state_dict() if is_parallel else embedding_network.state_dict(),
		'classification_state_dict': classification_network.module.state_dict() if is_parallel else classification_network.state_dict(),
		'optimizer': optimizer.state_dict(),
		'epoch': epoch,
		'train_loss_history': train_loss_history,
		'val_loss_history': val_loss_history,
		'train_accuracy_history': train_accuracy_history,
		'val_accuracy_history': val_accuracy_history
	}

	state_dict_name = 'state_dict_{}_numframes{}_numpairs{}_pool{}_res{}_epoch{}.pkl'\
					.format(os.path.splitext(dataset)[0], num_frames_in_stack, num_pairs_per_example, use_pool*1, use_res*1, epoch)
	state_dict_path = os.path.join(project_dir, checkpoints_dir, state_dict_name)
	logging.info('Saving checkpoint "{}"...'.format(state_dict_path))
	torch.save(state_dict, state_dict_path)
	logging.info('Done.')

def load_checkpoint(embedding_network, classification_network, optimizer, checkpoint_file, project_dir, checkpoints_dir, device):
	# Note: Input model & optimizer should be pre-defined. This routine only updates their states.

	train_loss_history, val_loss_history = [], []
	train_accuracy_history, val_accuracy_history = [], []
	epoch_trained = 0

	state_dict_path = os.path.join(project_dir, checkpoints_dir, checkpoint_file)

	if os.path.isfile(state_dict_path):
		logging.info('Loading checkpoint "{}"...'.format(state_dict_path))
		state_dict = torch.load(state_dict_path)

		# Extract last trained epoch from checkpoint file
		epoch_trained = int(os.path.splitext(checkpoint_file)[0].split('_')[-1])
		assert epoch_trained == state_dict['epoch']

		embedding_network.load_state_dict(state_dict['embedding_state_dict'])
		classification_network.load_state_dict(state_dict['classification_state_dict'])
		optimizer.load_state_dict(state_dict['optimizer'])
		train_loss_history = state_dict['train_loss_history']
		val_loss_history = state_dict['val_loss_history']
		train_accuracy_history = state_dict['train_accuracy_history']
		val_accuracy_history = state_dict['val_accuracy_history']

		for state in optimizer.state.values():
			for k, v in state.items():
				if isinstance(v, torch.Tensor):
					state[k] = v.to(device)

		logging.info('Successfully loaded checkpoint "{}".'.format(state_dict_path))

	else:
		raise FileNotFoundError('No checkpoint found at "{}"!'.format(state_dict_path))

	return embedding_network, classification_network, optimizer, train_loss_history, val_loss_history, \
			train_accuracy_history, val_accuracy_history, epoch_trained


class EarlyStopping(object):
	'''
	Implements early stopping in PyTorch
	Source: https://gist.github.com/stefanonardo/693d96ceb2f531fa05db530f3e21517d
	'''

	def __init__(self, mode='min', min_delta=0, patience=10):
		self.mode = mode
		self.min_delta = min_delta
		self.patience = patience
		self.best = None
		self.num_bad_epochs = 0
		self.is_better = None
		self._init_is_better(mode, min_delta)

		if patience == 0:
			self.is_better = lambda a, b: True
			self.step = lambda a: False

	def step(self, metrics):
		if self.best is None:
			self.best = metrics
			return False

		if np.isnan(metrics):
			return True

		if self.is_better(metrics, self.best):
			self.num_bad_epochs = 0
			self.best = metrics
		else:
			self.num_bad_epochs += 1

		if self.num_bad_epochs >= self.patience:
			return True
		return False

	def _init_is_better(self, mode, min_delta):
		if mode not in {'minimize', 'maximize'}:
			raise ValueError('mode "{}" is unknown!'.format(mode))
		if mode == 'minimize':
			self.is_better = lambda a, best: a < best - min_delta
		elif mode == 'maximize':
			self.is_better = lambda a, best: a > best + min_delta
