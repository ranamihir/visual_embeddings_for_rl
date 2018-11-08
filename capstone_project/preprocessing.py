import numpy as np
import pandas as pd
import os
import pickle
import logging

from sklearn.model_selection import train_test_split
import torch
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

from capstone_project.utils import imshow, save_object, load_object


class MovingMNISTDataset(Dataset):
	def __init__(self, data, time_buckets, num_frames_in_stack=2, size=300000, transforms=None):
		self.data = data
		self.size = size
		self.num_frames_in_stack = num_frames_in_stack
		self.time_buckets_dict = self._get_time_buckets_dict(time_buckets)
		self._check_data()
		self.differences_dict = self._get_frame_differences_dict()
		self.transforms = transforms

	def __getitem__(self, index):
		video_idx = np.random.choice(len(self.data))
		difference = np.random.choice(list(self.time_buckets_dict.keys()))

		(x1, x2), y, difference, frame_numbers = self._get_sample_at_difference(video_idx, difference)

		if self.transforms:
			x1 = torch.stack([self.transforms(x[:,:,np.newaxis]) for x in x1], dim=0).squeeze(1)
			x2 = torch.stack([self.transforms(x[:,:,np.newaxis]) for x in x2], dim=0).squeeze(1)

		y, difference = torch.from_numpy(y), torch.from_numpy(difference)
		frame1, frame2 = torch.from_numpy(frame_numbers)

		return x1, x2, y, difference, (frame1, frame2)

	def __len__(self):
		return self.size

	def _check_data(self):
		sequence_length = self.data.shape[1]
		max_frame_diff = max(self.time_buckets_dict.keys())
		assert max_frame_diff <= sequence_length-self.num_frames_in_stack, \
			'Cannot have difference of {} when sequence length is {} and number of \
			stacked frames are {}'.format(max_frame_diff, sequence_length, self.num_frames_in_stack)

	def _get_time_buckets_dict(self, time_buckets):
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

	def _get_frame_differences_dict(self):
		'''
		Returns a dict with the key as the time difference between the frames
		and the value as a list of tuples (start_frame, end_frame) containing
		all the pair of frames with that diff in time
		'''
		logging.info('Getting frame differences dictionary...')
		sequence_length = self.data.shape[1]
		max_frame_diff = max(self.time_buckets_dict.keys())

		differences_dict = {}
		differences = range(max_frame_diff+1)
		for diff in differences:
			start_frame = self.num_frames_in_stack-1
			end_frame = start_frame+diff
			while end_frame <= sequence_length-1:
				differences_dict.setdefault(diff, []).append(tuple((start_frame, end_frame)))
				start_frame += 1
				end_frame += 1
		logging.info('Done.')
		return differences_dict

	def _get_sample_at_difference(self, video_idx, difference):
		'''
		Returns samples by selecting the list of tuples for the associated time difference,
		sampling the num of pairs per video example, and finally returning the (stacked) image pairs,
		their associated buckets (classes), frame difference, and last frame numbers for each pair (tuple)
		'''
		video = self.data[video_idx]
		candidates = self.differences_dict[difference]
		pair_idx = np.random.choice(len(candidates))
		image1_last_frame, image2_last_frame = candidates[pair_idx]

		image1_frames = list(range(image1_last_frame-self.num_frames_in_stack+1, image1_last_frame+1))
		image2_frames = list(range(image2_last_frame-self.num_frames_in_stack+1, image2_last_frame+1))
		image_pair = np.array([video[image1_frames], video[image2_frames]])

		bucket = np.array(self.time_buckets_dict[difference])

		frame_numbers = np.array([image1_last_frame, image2_last_frame])

		return image_pair, bucket, np.array(difference), frame_numbers


class AtariDataset(Dataset):
	def __init__(self, data, time_buckets, num_frames_in_stack=4, size=300000, transforms=None):
		self.data = data
		self.size = size
		self.num_frames_in_stack = num_frames_in_stack
		self.time_buckets_dict = self._get_time_buckets_dict(time_buckets)
		self._check_data()
		self.differences_dict = self._get_frame_differences_dict()
		self.transforms = transforms

	def __getitem__(self, index):
		video_idx = np.random.choice(len(self.data))
		difference = np.random.choice(list(self.time_buckets_dict.keys()))

		(x1, x2), y, difference, frame_numbers = self._get_sample_at_difference(video_idx, difference)

		if self.transforms:
			x1 = torch.stack([self.transforms(x[:,:,np.newaxis]) for x in x1], dim=0).squeeze(1)
			x2 = torch.stack([self.transforms(x[:,:,np.newaxis]) for x in x2], dim=0).squeeze(1)

		y, difference = torch.from_numpy(y), torch.from_numpy(difference)
		frame1, frame2 = torch.from_numpy(frame_numbers)

		return x1, x2, y, difference, (frame1, frame2)

	def __len__(self):
		return self.size

	def _check_data(self):
		sequence_length = self.data.shape[1]
		max_frame_diff = max(self.time_buckets_dict.keys())
		assert max_frame_diff <= sequence_length-self.num_frames_in_stack, \
			'Cannot have difference of {} when sequence length is {} and number of \
			stacked frames are {}'.format(max_frame_diff, sequence_length, self.num_frames_in_stack)

	def _get_time_buckets_dict(self, time_buckets):
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

	def _get_frame_differences_dict(self):
		'''
		Returns a dict with the key as the time difference between the frames
		and the value as a list of tuples (start_frame, end_frame) containing
		all the pair of frames with that diff in time
		'''
		logging.info('Getting frame differences dictionary...')
		sequence_length = self.data.shape[1]
		max_frame_diff = max(self.time_buckets_dict.keys())

		differences_dict = {}
		differences = range(max_frame_diff+1)
		for diff in differences:
			start_frame, end_frame = 0, diff
			while end_frame <= sequence_length-1:
				differences_dict.setdefault(diff, []).append(tuple((start_frame, end_frame)))
				start_frame += 1
				end_frame += 1
		logging.info('Done.')
		return differences_dict

	def _get_sample_at_difference(self, video_idx, difference):
		'''
		Returns samples by selecting the list of tuples for the associated time difference,
		sampling the num of pairs per video example, and finally returning the (stacked) image pairs,
		their associated buckets (classes), frame difference, and last frame numbers for each pair (tuple)
		'''
		video = self.data[video_idx]
		candidates = self.differences_dict[difference]
		pair_idx = np.random.choice(len(candidates))
		image1_frame_idx, image2_frame_idx = candidates[pair_idx]
		image_pair_idxs = np.array([image1_frame_idx, image2_frame_idx])
		image_pair = np.array([video[image1_frame_idx], video[image2_frame_idx]])

		bucket = np.array(self.time_buckets_dict[difference])

		return image_pair, bucket, np.array(difference), image_pair_idxs

def generate_online_dataloader(project_dir, data_dir, plots_dir, dataset, dataset_size, dataset_type, \
							time_buckets, batch_size, num_frames_in_stack=2, ext='.npy', \
							transforms=None, data_max=None, data_min=None):
	if dataset_type == 'train':
		data, data_max, data_min = load_data(project_dir, data_dir, dataset, dataset_type, ext)
	else:
		data = load_data(project_dir, data_dir, dataset, dataset_type, ext, data_max, data_min)

	if 'pong' in dataset:
		IS_STACKED_DATA = 1
		assert num_frames_in_stack == data.shape[-3], \
			'NUM_FRAMES_IN_STACK (={}) must match number of stacked images in stacked dataset (={})!'\
			.format(num_frames_in_stack, data.shape[-3])
	elif 'mnist' in dataset or 'moving_bars' in dataset:
		IS_STACKED_DATA = 0
	else:
		raise ValueError('Unknown dataset name "{}" passed!'.format(dataset))

	if dataset_type == 'train':
		transforms, mean, std = get_normalize_transform(data, num_frames_in_stack)
		imshow(data, mean, std, project_dir, plots_dir, dataset)

	logging.info('Generating {} data loader...'.format(dataset_type))
	if IS_STACKED_DATA:
		dataset = AtariDataset(data, time_buckets, num_frames_in_stack, \
							dataset_size, transforms=transforms)
	else:
		dataset = MovingMNISTDataset(data, time_buckets, num_frames_in_stack, \
									dataset_size, transforms=transforms)

	shuffle = 1 if dataset_type == 'train' else False
	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
	logging.info('Done.')

	if dataset_type == 'train':
		return dataloader, transforms, data_max, data_min
	return dataloader


class OfflineMovingMNISTDataset(Dataset):
	def __init__(self, X, y, differences, frame_numbers, transforms=None):
		self.X = X
		self.y = torch.from_numpy(y)
		self.differences = torch.from_numpy(differences)
		self.frame_numbers = torch.from_numpy(frame_numbers)
		self.transforms = transforms

	def __getitem__(self, index):
		x1, x2 = self.X[index]
		y = self.y[index]
		difference = self.differences[index]
		frame1, frame2 = self.frame_numbers[index]

		if self.transforms:
			x1 = torch.stack([self.transforms(x[:,:,np.newaxis]) for x in x1], dim=0).squeeze(1)
			x2 = torch.stack([self.transforms(x[:,:,np.newaxis]) for x in x2], dim=0).squeeze(1)

		return x1, x2, y, difference, (frame1, frame2)

	def __len__(self):
		return len(self.y)

def generate_all_offline_dataloaders(project_dir, data_dir, plots_dir, filename, time_buckets, batch_size, \
						num_pairs_per_example=5, num_frames_in_stack=2, ext='.npy', force=False):

	data_path = os.path.join(project_dir, data_dir, '{}_{}_{}_{}.pkl')
	train_path = data_path.format(filename, 'train', num_frames_in_stack, num_pairs_per_example)
	val_path = data_path.format(filename, 'val', num_frames_in_stack, num_pairs_per_example)
	test_path = data_path.format(filename, 'test', num_frames_in_stack, num_pairs_per_example)

	if not force and os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path):
		logging.info('Found all data sets on disk.')
		dataloaders = []
		for dataset_type in ['train', 'val', 'test']:
			logging.info('Loading {} data set...'.format(dataset_type))
			dataset = load_object(data_path.format(filename, dataset_type, num_frames_in_stack, num_pairs_per_example))
			shuffle = 1 if dataset_type == 'train' else False
			dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
			dataloaders.append(dataloader)
			logging.info('Done.')

	else:
		logging.info('Did not find at least one data set on disk. Creating all 3...')

		data_dict = {}
		data_dict['train'], data_max, data_min = load_data(project_dir, data_dir, filename, 'train', ext)
		for dataset_type in ['val', 'test']:
			data_dict[dataset_type] = load_data(project_dir, data_dir, filename, dataset_type, ext, data_max, data_min)

		sequence_length = data_dict['train'].shape[1]
		max_frame_diff = np.hstack([bucket for bucket in time_buckets]).max()
		assert max_frame_diff <= sequence_length-num_frames_in_stack, \
			'Cannot have difference of {} when sequence length is {} and number of \
			stacked frames are {}'.format(max_frame_diff, sequence_length, num_frames_in_stack)

		# Calculate mean, std from train data, and normalize data
		transforms, mean, std = get_normalize_transform(data_dict['train'], num_frames_in_stack)
		imshow(data_dict['train'], mean, std, project_dir, plots_dir, filename)

		time_buckets_dict = get_time_buckets_dict(time_buckets)
		differences_dict = get_frame_differences_dict(sequence_length, max_frame_diff, num_frames_in_stack)

		dataloaders = []
		for dataset_type in ['train', 'val', 'test']:
			logging.info('Generating {} data set...'.format(dataset_type))
			stacked_img_pairs, target_buckets = np.array([]), np.array([])
			target_differences, target_frame_numbers = np.array([]), np.array([])

			for difference in range(max_frame_diff+1):
				img_pairs, buckets, differences, frame_numbers = get_samples_at_difference(data_dict[dataset_type], \
																difference, differences_dict, num_pairs_per_example, \
																num_frames_in_stack, time_buckets_dict)
				stacked_img_pairs = np.vstack((stacked_img_pairs, img_pairs)) if stacked_img_pairs.size else img_pairs
				target_buckets = np.append(target_buckets, buckets)
				target_differences = np.append(target_differences, differences)
				target_frame_numbers = np.vstack((target_frame_numbers, frame_numbers)) if target_frame_numbers.size else frame_numbers

			dataset = OfflineMovingMNISTDataset(stacked_img_pairs, target_buckets, target_differences, target_frame_numbers, transforms=transforms)
			logging.info('Done. Dumping {} data set to disk...'.format(dataset_type))
			save_object(dataset, data_path.format(filename, dataset_type, num_frames_in_stack, num_pairs_per_example))
			logging.info('Done.')

			shuffle = True if dataset_type == 'train' else False
			dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
			dataloaders.append(dataloader)

	return dataloaders

def load_data(project_dir, data_dir, filename, dataset_type, ext, \
			data_max=None, data_min=None):
	filename = '{}_{}{}'.format(filename, dataset_type, ext)
	logging.info('Loading "{}"...'.format(filename))
	file_path = os.path.join(project_dir, data_dir, filename)

	if ext == '.npy':
		data = np.load(file_path)
	elif ext == '.pkl':
		data = load_object(file_path)
	else:
		raise ValueError('Unknown file extension "{}"!'.format(ext))
	logging.info('Done.')

	# Bring data in [0,255]
	if dataset_type == 'train' and data_max == None and data_min == None:
		data_min, data_max = data.min(), data.max()
		data = 255*(data - data_min)/(data_max-data_min)
		return data.astype(np.uint8), data_max, data_min
	else:
		data = 255*(data - data_min)/(data_max-data_min)
		return data.astype(np.uint8)

def get_normalize_transform(data, num_frames_in_stack):
	# Calculate mean, std from train data, and normalize
	mean = np.mean(data)
	std = np.std(data)

	normalize = transforms.Compose([
		transforms.ToTensor(),
	    transforms.Normalize((mean,), (std,))
	])

	return normalize, mean, std

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
