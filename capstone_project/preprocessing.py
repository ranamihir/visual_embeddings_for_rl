import numpy as np
import pandas as pd
import os
import pickle
import logging

from sklearn.model_selection import train_test_split
import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

from capstone_project.utils import imshow, save_object, load_object


class MovingMNISTDataset(Dataset):
	# TODO: write function for implementing transforms
	def __init__(self, X, y, differences, frame_numbers, transforms=None):
		self.X = Variable(X)
		self.y = Variable(y)
		self.differences = Variable(differences)
		self.frame_numbers = Variable(frame_numbers)
		self.transforms = transforms

	def __getitem__(self, index):
		# if self.transforms is not None:
		#     data = self.transforms(data)

		# Load data and get label
		x1, x2 = self.X[index]
		y = self.y[index]
		difference = self.differences[index]
		frame1, frame2 = self.frame_numbers[index]
		return x1, x2, y, difference, (frame1, frame2)

	def __len__(self):
		return len(self.y)

def generate_dataloaders(project_dir, data_dir, plots_dir, filename, time_buckets, batch_size, num_passes_for_generation=1, \
						num_pairs_per_example=1, num_frames_in_stack=2, val_size=0.2, test_size=0.2, force=False):

	filename_without_ext, ext = os.path.splitext(filename)
	data_path = os.path.join(project_dir, data_dir, '{}_{}_{}_{}.pkl')
	train_path = data_path.format(filename_without_ext, 'train', num_frames_in_stack, num_pairs_per_example)
	val_path = data_path.format(filename_without_ext, 'val', num_frames_in_stack, num_pairs_per_example)
	test_path = data_path.format(filename_without_ext, 'test', num_frames_in_stack, num_pairs_per_example)

	if not force and os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path):
		logging.info('Found all data sets on disk.')
		data_loaders = []
		for dataset_type in ['train', 'val', 'test']:
			logging.info('Loading {} data set...'.format(dataset_type))
			dataset = load_object(data_path.format(filename_without_ext, dataset_type, num_frames_in_stack, num_pairs_per_example))
			data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True if dataset_type == 'train' else False, num_workers=2)
			data_loaders.append(data_loader)
			logging.info('Done.')

	else:
		logging.info('Did not find at least one data set on disk. Creating all 3...')

		logging.info('Loading "{}"...'.format(filename))
		data = load_data(project_dir, data_dir, filename)
		logging.info('Done.')

		sequence_length = data.shape[1]
		max_frame_diff = np.hstack([bucket for bucket in time_buckets]).max()
		assert max_frame_diff <= sequence_length-num_frames_in_stack, \
			'Cannot have difference of {} when sequence length is {} and number of \
			stacked frames are {}'.format(max_frame_diff, sequence_length, num_frames_in_stack)

		logging.info('Splitting data set into train, val, and test sets...')
		data_dict = split_data(data, val_size, test_size, project_dir, data_dir)
		data = None # Free data
		logging.info('Done.')

		mean, std = 0, 1 # Calculate mean, std from train data
		# TODO: Normalize data
		# mean = np.mean(dataset)
		# std = np.std(dataset)
		# dataset = (dataset - mean)/std
		# data_transform = transforms.Compose([
		#     transforms.ToTensor(),
		#     transforms.Normalize((mean,), (std,))
		# ])

		imshow(data_dict['train'], mean, std, project_dir, plots_dir)

		time_buckets_dict = get_time_buckets_dict(time_buckets)
		differences_dict = get_frame_differences_dict(sequence_length, max_frame_diff, num_frames_in_stack)

		data_loaders = []
		for dataset_type in ['train', 'val', 'test']:
			logging.info('Generating {} data set...'.format(dataset_type))
			stacked_img_pairs, target_buckets = np.array([]), np.array([])
			target_differences, target_frame_numbers = np.array([]), np.array([])
			for i in range(num_passes_for_generation):
				logging.info('Making pass {} through data...'.format(i+1))
				for difference in range(max_frame_diff+1):
					img_pairs, buckets, differences, frame_numbers = get_samples_at_difference(data_dict[dataset_type], \
																	difference, differences_dict, num_pairs_per_example, \
																	num_frames_in_stack, time_buckets_dict)
					stacked_img_pairs = np.vstack((stacked_img_pairs, img_pairs)) if stacked_img_pairs.size else img_pairs
					target_buckets = np.append(target_buckets, buckets)
					target_differences = np.append(target_differences, differences)
					target_frame_numbers = np.vstack((target_frame_numbers, frame_numbers)) if target_frame_numbers.size else frame_numbers
				logging.info('Done.')

			stacked_img_pairs, target_buckets = torch.from_numpy(stacked_img_pairs), torch.from_numpy(target_buckets)
			target_differences, target_frame_numbers = torch.from_numpy(target_differences), torch.from_numpy(target_frame_numbers)
			dataset = MovingMNISTDataset(stacked_img_pairs, target_buckets, target_differences, target_frame_numbers, transforms=None)
			logging.info('Done. Dumping {} data set to disk...'.format(dataset_type))
			save_object(dataset, data_path.format(filename_without_ext, dataset_type, num_frames_in_stack, num_pairs_per_example))
			logging.info('Done.')

			data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
			data_loaders.append(data_loader)

	return data_loaders

def load_data(project_dir, data_dir, filename):
	filename = os.path.join(project_dir, data_dir, filename)
	data = np.load(filename)
	data = data.swapaxes(0,1)
	return data

def split_data(data, val_size, test_size, project_dir, data_dir):
	num_test = int(np.floor(test_size*len(data)))
	num_train_val = len(data) - num_test
	num_val = int(np.floor(num_train_val*val_size/(1 - test_size)))

	train_data, test_data = train_test_split(data, test_size=num_test)
	train_data, val_data = train_test_split(train_data, test_size=num_val)

	data_dict = {
		'train': train_data,
		'val': val_data,
		'test': test_data
	}

	return data_dict

def get_time_buckets_dict(time_buckets):
	'''
	Returns a dict, with the time diff
	as its key and the target class (0-indexed) for it as its value
	'''
	logging.info('Getting time buckets dictionary...')
	bucket_idx = 0
	buckets_dict = {}
	for bucket in time_buckets:
		for time in bucket:
			buckets_dict[time] = bucket_idx
		bucket_idx += 1
	logging.info('Done.')
	return buckets_dict

def get_frame_differences_dict(sequence_length, max_frame_diff, num_frames_in_stack):
	'''
	Returns a dict with the key as the time difference between the frames
	and the value as a list of tuples (start_frame, end_frame) containing
	all the pair of frames with that diff in time
	'''
	logging.info('Getting frame differences dictionary...')
	differences_dict = {}
	differences = range(max_frame_diff+1)
	for diff in differences:
		start_frame = num_frames_in_stack-1
		end_frame = start_frame+diff
		while end_frame <= sequence_length-1:
			differences_dict.setdefault(diff, []).append(tuple((start_frame, end_frame)))
			start_frame += 1
			end_frame += 1
	logging.info('Done.')
	return differences_dict

def get_samples_at_difference(data, difference, differences_dict, num_pairs_per_example, num_frames_in_stack, time_buckets_dict):
	'''
	Returns samples by selecting the list of tuples for the associated time difference,
	sampling the num of pairs per video example, and finally returning the (stacked) image pairs,
	their associated buckets (classes), frame difference, and last frame numbers for each pair (tuple)
	'''
	logging.info('Getting all pairs with a frame difference of {}...'.format(difference))
	img_pairs, target_buckets, differences, frame_numbers = [], [], [], []
	candidates = differences_dict[difference]
	np.random.seed(1337)
	idx_pairs = np.random.choice(len(candidates), size=num_pairs_per_example)
	for row in data:
		for idx_pair in idx_pairs:
			target1_last_frame, target2_last_frame = candidates[idx_pair]
			target1_frames = list(range(target1_last_frame-num_frames_in_stack+1, target1_last_frame+1))
			target2_frames = list(range(target2_last_frame-num_frames_in_stack+1, target2_last_frame+1))
			img_pairs.append([row[target1_frames], row[target2_frames]])
			bucket = time_buckets_dict[difference]
			target_buckets.append(bucket)
			differences.append(difference)
			frame_numbers.append(tuple((target1_last_frame, target2_last_frame)))
	logging.info('Done.')
	return np.array(img_pairs), np.array(target_buckets), np.array(differences), np.array(frame_numbers)
