import numpy as np
import pandas as pd
import os
import pickle
import h5py
import logging

from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import DataLoader

from capstone_project.datasets import *
from capstone_project.utils import imshow, plot_video, save_object, load_object

def generate_online_dataloader(project_dir, data_dir, plots_dir, dataset_name, dataset_size, dataset_type, \
							time_buckets, batch_size, num_frames_in_stack=2, ext='.npy', transforms=None):
	data = load_data(project_dir, data_dir, dataset_name, dataset_type, ext)

	if len(data.shape) > 3:
		if len(data.shape) == 5 and data.shape[-3] > 1:
			IS_STACKED_DATA = 1
			assert num_frames_in_stack == data.shape[-3], \
				'NUM_FRAMES_IN_STACK (={}) must match number of stacked images in stacked dataset (={})!'\
				.format(num_frames_in_stack, data.shape[-3])
		else:
			assert len(data.shape) == 4, 'Unknown input data shape "{}"'.format(data.shape)
			IS_STACKED_DATA = 0

		# Normalize data and create dataloaders
		if dataset_type == 'train':
			transforms, mean, std = get_normalize_transform(data, num_frames_in_stack)
			imshow(data, mean, std, project_dir, plots_dir, dataset_name)

		logging.info('Generating {} data loader...'.format(dataset_type))
		if IS_STACKED_DATA:
			dataset = AtariDataset(data, time_buckets, num_frames_in_stack, \
								dataset_size, transforms=transforms)
		else:
			dataset = FixedMovingMNISTDataset(data, time_buckets, num_frames_in_stack, \
										dataset_size, transforms=transforms)
	else:
		assert len(data.shape) == 3, 'Unknown input data shape "{}"'.format(data.shape)

		logging.info('Generating {} data loader...'.format(dataset_type))
		video_generator = RandomMovingMNISTVideoGenerator(data)
		dataset = RandomMovingMNISTDataset(video_generator, time_buckets, num_frames_in_stack, \
											dataset_size)
		video = video_generator.__getitem__()
		imshow(video, 0, 1, project_dir, plots_dir, dataset_name)
		plot_video(video, project_dir, plots_dir, dataset_name)
		transforms = None

	shuffle = 1 if dataset_type == 'train' else False
	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
	logging.info('Done.')

	if dataset_type == 'train':
		return dataloader, transforms
	return dataloader

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
		for dataset_type in ['train', 'val', 'test']:
			data_dict[dataset_type] = load_data(project_dir, data_dir, filename, dataset_type, ext)

		sequence_length = data_dict['train'].shape[1]
		max_frame_diff = np.hstack([bucket for bucket in time_buckets]).max()
		assert max_frame_diff <= sequence_length-num_frames_in_stack, \
			'Cannot have difference of {} when sequence length is {} and number of \
			stacked frames are {}'.format(max_frame_diff, sequence_length, num_frames_in_stack)

		# Normalize data
		transforms, mean, std = get_normalize_transform(data_dict['train'], num_frames_in_stack)
		imshow(data_dict['train'], project_dir, plots_dir, filename)

		time_buckets_dict = get_time_buckets_dict(time_buckets)
		differences_dict = get_candidates_differences_dict(sequence_length, max_frame_diff, num_frames_in_stack)

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
				target_frame_numbers = np.vstack((target_frame_numbers, frame_numbers)) \
										if target_frame_numbers.size else frame_numbers

			dataset = OfflineMovingMNISTDataset(stacked_img_pairs, target_buckets, target_differences, \
												target_frame_numbers, transforms=transforms)
			logging.info('Done. Dumping {} data set to disk...'.format(dataset_type))
			save_object(dataset, data_path.format(filename, dataset_type, num_frames_in_stack, num_pairs_per_example))
			logging.info('Done.')

			shuffle = True if dataset_type == 'train' else False
			dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
			dataloaders.append(dataloader)

	return dataloaders

def load_data(project_dir, data_dir, filename, dataset_type, ext):
	filename = '{}_{}{}'.format(filename, dataset_type, ext)
	logging.info('Loading "{}"...'.format(filename))
	file_path = os.path.join(project_dir, data_dir, filename)

	if ext == '.npy':
		data = np.load(file_path)
	elif ext == '.pkl':
		data = load_object(file_path)
	elif ext == '.h5':
		with h5py.File(file_path) as f:
			data = np.array(f['inputs'])
	else:
		raise ValueError('Unknown file extension "{}"!'.format(ext))
	logging.info('Done.')

	return data

def get_normalize_transform(data, num_frames_in_stack):
	# Calculate mean, std from train data, and normalize
	mean = np.mean(data)
	std = np.std(data)

	normalize = transforms.Compose([
		transforms.Normalize((mean,)*num_frames_in_stack, (std,)*num_frames_in_stack)
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

def get_candidates_differences_dict(sequence_length, max_frame_diff, num_frames_in_stack):
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

def get_samples_at_difference(data, difference, differences_dict, num_pairs_per_example, \
							num_frames_in_stack, time_buckets_dict):
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
			target1_frames = range(target1_last_frame-num_frames_in_stack+1, target1_last_frame+1)
			target2_frames = range(target2_last_frame-num_frames_in_stack+1, target2_last_frame+1)
			img_pairs.append([row[target1_frames], row[target2_frames]])
			bucket = time_buckets_dict[difference]
			target_buckets.append(bucket)
			differences.append(difference)
			frame_numbers.append(tuple((target1_last_frame, target2_last_frame)))
	logging.info('Done.')
	return np.array(img_pairs), np.array(target_buckets), np.array(differences), np.array(frame_numbers)

def split_data(data, val_size, test_size, project_dir, data_dir):
	if 0. <= val_size <= 1. and 0. <= test_size <= 1.:
		num_test = int(np.floor(test_size*len(data)))
		num_train_val = len(data) - num_test
		num_val = int(np.floor(num_train_val*val_size/(1 - test_size)))
	else:
		num_val, num_test = val_size, test_size

	train_data, test_data = train_test_split(data, test_size=num_test, random_state=1337)
	train_data, val_data = train_test_split(train_data, test_size=num_val, random_state=1337)

	data_dict = {
		'train': train_data,
		'val': val_data,
		'test': test_data
	}

	return data_dict
