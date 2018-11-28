import numpy as np
import pandas as pd
import os
import h5py
import logging

from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import DataLoader

from capstone_project.datasets import *
from capstone_project.utils import imshow, plot_video, save_object, load_object

def generate_online_dataloader(project_dir, data_dir, plots_dir, dataset_type, \
                               dataset_name, dataset_size, data_type, time_buckets, \
                               model, batch_size, num_frames_in_stack=2, \
                               num_channels=1, ext='.npy', flatten=False, \
                               transforms=None, force=False):
    assert dataset_type in ['maze', 'random_mmnist', 'fixed_mmnist'], \
        'Unknown dataset type "{}" passed.'.format(dataset_type)

    logging.info('Generating {} data loader...'.format(data_type))

    # Maze Dataset
    if dataset_type == 'maze':
        data = load_maze_data(project_dir, data_dir, dataset_name, data_type, flatten, force)
        assert len(data[0].shape) == 4, 'Unknown input data shape "{}"'.format(data.shape)
        assert model in ['cnn', 'emb-cnn1', 'emb-cnn2', 'rel'], \
            'Unknown model name "{}" passed.'.format(model)

        dataset = MazeDataset(data, time_buckets, num_channels, dataset_size, \
                              return_embedding=False if model == 'cnn' else True)
        transforms = None

    # Fixed Moving MNIST Dataset
    elif dataset_type == 'fixed_mmnist':
        data = load_data(project_dir, data_dir, dataset_name, data_type, ext)
        assert len(data.shape) == 4, 'Unknown input data shape "{}"'.format(data.shape)

        # Normalize data and create dataloaders
        if data_type == 'train':
            transforms, mean, std = get_normalize_transform(data, num_frames_in_stack)
            imshow(data, mean, std, project_dir, plots_dir, dataset_name)

        dataset = FixedMovingMNISTDataset(data, time_buckets, num_frames_in_stack, \
                                    dataset_size, transforms=transforms)

    # Random Moving MNIST Dataset
    elif dataset_type == 'random_mmnist':
        data = load_data(project_dir, data_dir, dataset_name, data_type, ext)
        assert len(data.shape) == 3, 'Unknown input data shape "{}"'.format(data.shape)

        video_generator = RandomMovingMNISTVideoGenerator(data)
        dataset = RandomMovingMNISTDataset(video_generator, time_buckets, num_frames_in_stack, \
                                            dataset_size)
        video = video_generator.__getitem__()
        imshow(video, 0, 1, project_dir, plots_dir, dataset_name)
        plot_video(video, project_dir, plots_dir, dataset_name)
        transforms = None

    shuffle = 1 if data_type == 'train' else False
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
    logging.info('Done.')

    if data_type == 'train':
        return dataloader, transforms
    return dataloader

def generate_all_offline_dataloaders(project_dir, data_dir, plots_dir, filename, time_buckets, \
                                    batch_size, num_pairs_per_example=5, num_frames_in_stack=2, \
                                    ext='.npy', force=False):

    data_path = os.path.join(project_dir, data_dir, '{}_{}_{}_{}.pkl')
    train_path = data_path.format(filename, 'train', num_frames_in_stack, num_pairs_per_example)
    val_path = data_path.format(filename, 'val', num_frames_in_stack, num_pairs_per_example)
    test_path = data_path.format(filename, 'test', num_frames_in_stack, num_pairs_per_example)

    if not force and os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path):
        logging.info('Found all data sets on disk.')
        dataloaders = []
        for data_type in ['train', 'val', 'test']:
            logging.info('Loading {} data set...'.format(data_type))
            dataset = load_object(data_path.format(filename, data_type, num_frames_in_stack, \
                                                   num_pairs_per_example))
            shuffle = 1 if data_type == 'train' else False
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
            dataloaders.append(dataloader)
            logging.info('Done.')

    else:
        logging.info('Did not find at least one data set on disk. Creating all 3...')

        data_dict = {}
        for data_type in ['train', 'val', 'test']:
            data_dict[data_type] = load_data(project_dir, data_dir, filename, data_type, ext)

        sequence_length = data_dict['train'].shape[1]
        max_frame_diff = np.hstack([bucket for bucket in time_buckets]).max()
        assert max_frame_diff <= sequence_length-num_frames_in_stack, \
            'Cannot have difference of {} when sequence length is {} and number of \
            stacked frames are {}'.format(max_frame_diff, sequence_length, num_frames_in_stack)

        # Normalize data
        transforms, mean, std = get_normalize_transform(data_dict['train'], num_frames_in_stack)
        imshow(data_dict['train'], project_dir, plots_dir, filename)

        time_buckets_dict = get_time_buckets_dict(time_buckets)
        differences_dict = get_candidates_differences_dict(sequence_length, max_frame_diff, \
                                                           num_frames_in_stack)

        dataloaders = []
        for data_type in ['train', 'val', 'test']:
            logging.info('Generating {} data set...'.format(data_type))
            stacked_img_pairs, target_buckets = np.array([]), np.array([])
            target_differences, target_frame_numbers = np.array([]), np.array([])

            for difference in range(max_frame_diff+1):
                img_pairs, buckets, differences, frame_numbers = \
                    get_samples_at_difference(data_dict[data_type], difference, differences_dict, \
                                              num_pairs_per_example, num_frames_in_stack, time_buckets_dict)
                stacked_img_pairs = np.vstack((stacked_img_pairs, img_pairs)) if stacked_img_pairs.size \
                                                                              else img_pairs
                target_buckets = np.append(target_buckets, buckets)
                target_differences = np.append(target_differences, differences)
                target_frame_numbers = np.vstack((target_frame_numbers, frame_numbers)) \
                                        if target_frame_numbers.size else frame_numbers

            dataset = OfflineMovingMNISTDataset(stacked_img_pairs, target_buckets, target_differences, \
                                                target_frame_numbers, transforms=transforms)
            logging.info('Done. Dumping {} data set to disk...'.format(data_type))
            save_object(dataset, data_path.format(filename, data_type, num_frames_in_stack, \
                                                  num_pairs_per_example))
            logging.info('Done.')

            shuffle = True if data_type == 'train' else False
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
            dataloaders.append(dataloader)

    return dataloaders

def load_data(project_dir, data_dir, filename, data_type, ext):
    filename_dataset = '{}_{}{}'.format(filename, data_type, ext)
    logging.info('Loading "{}"...'.format(filename_dataset))
    file_path = os.path.join(project_dir, data_dir, filename_dataset)

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

def load_maze_data(project_dir, data_dir, filename, data_type, flatten=False, force=False):
    filename_dataset = '{}_{}.h5'.format(filename, data_type)
    file_path = os.path.join(project_dir, data_dir, filename_dataset)
    if not force and os.path.exists(file_path):
        logging.info('Loading "{}"...'.format(file_path))
        with h5py.File(file_path, 'r') as f:
            data = [f[key][:] for key in f.keys()]
        logging.info('Done.')
    else:
        data = split_and_dump_maze_data(project_dir, data_dir, filename, data_type)

    seq_lens = np.array([maze.shape[0] for maze in data])
    logging.info('Min/Max/Avg/Total sequence length in {} data: '\
                 '{:.0f}/{:.0f}/{:.0f}/{:.0f}'.format(data_type, \
                 np.min(seq_lens), np.max(seq_lens), np.mean(seq_lens), np.sum(seq_lens)))

    if flatten:
        data = np.vstack([maze for maze in data])[np.newaxis,:]
    return data

def split_and_dump_maze_data(project_dir, data_dir, filename, data_type, val_size=0.2, test_size=0.2):
    file_path_in = os.path.join(project_dir, data_dir, '{}.h5'.format(filename))
    logging.info('Loading "{}"...'.format(file_path_in))
    with h5py.File(file_path_in, 'r') as f_in:
        all_data, keys = [], []
        for key in f_in.keys():
            all_data.append(f_in[key][:])
            keys.append(key)
    logging.info('Done.')

    # Randomly shuffle data set
    p = np.random.RandomState(1337).permutation(range(len(all_data)))
    all_data = [all_data[i] for i in p]
    keys = [keys[i] for i in p]

    n = len(all_data)
    num_test = int(test_size*len(all_data))
    num_val = int(val_size*len(all_data))
    start_idx = {
        'train': 0,
        'val': n-num_test-num_val,
        'test': -num_test
    }
    data_dict = {
        'train': all_data[:n-num_test-num_val],
        'val': all_data[n-num_test-num_val:n-num_test],
        'test': all_data[-num_test:]
    }

    file_path_out = os.path.join(project_dir, data_dir, '{}_{}.h5'.format(filename, data_type))
    logging.info('Dumping and returning {}...'.format(file_path_out))
    with h5py.File(file_path_out, 'w') as f_out:
        for i in range(len(data_dict[data_type])):
            # Retain the original keys when splitting into train/val/test
            f_out.create_dataset(keys[start_idx[data_type]+i], data=data_dict[data_type][i])
    logging.info('Done.')

    return data_dict[data_type]

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
