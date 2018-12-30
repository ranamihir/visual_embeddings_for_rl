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


def generate_online_dataloader(args, dataset_size, data_type, transforms=None):
    assert args.dataset_type in ['maze', 'random_mmnist', 'fixed_mmnist'], \
        'Unknown dataset type "{}" passed.'.format(args.dataset_type)

    logging.info('Generating {} data loader...'.format(data_type))

    # Maze Dataset
    if args.dataset_type == 'maze':
        data = load_maze_data(args, data_type)
        assert len(data[0].shape) == 4, 'Unknown input data shape "{}"'.format(data.shape)
        assert args.emb_model in ['cnn', 'emb-cnn1', 'emb-cnn2', 'rel'], \
            'Unknown embedding network model name "{}" passed.'.format(args.emb_model)

        dataset = MazeDataset(data, args.time_buckets, args.num_channels, dataset_size, \
                              return_indices=False if args.emb_model == 'cnn' else True)
        transforms = None

    # Fixed Moving MNIST Dataset
    elif args.dataset_type == 'fixed_mmnist':
        data = load_data(args, data_type)
        assert len(data.shape) == 4, 'Unknown input data shape "{}"'.format(data.shape)

        # Normalize data and create dataloaders
        if data_type == 'train':
            transforms, mean, std = get_normalize_transform(args, data)
            imshow(data, mean, std, args)

        dataset = FixedMovingMNISTDataset(data, args.time_buckets, args.num_frames, \
                                          dataset_size, transforms=transforms)

    # Random Moving MNIST Dataset
    elif args.dataset_type == 'random_mmnist':
        data = load_data(args, data_type)
        assert len(data.shape) == 3, 'Unknown input data shape "{}"'.format(data.shape)

        video_generator = RandomMovingMNISTVideoGenerator(data)
        dataset = RandomMovingMNISTDataset(video_generator, args.time_buckets, \
                                           args.num_frames, dataset_size)
        video = video_generator.__getitem__()
        imshow(video, 0, 1, args)
        plot_video(video, args)
        transforms = None

    shuffle = 1 if data_type == 'train' else False
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=4)
    logging.info('Done.')

    if data_type == 'train':
        return dataloader, transforms
    return dataloader

def generate_all_offline_dataloaders(args):

    data_path = os.path.join(args.project_dir, args.data_dir, '{}_{}_{}_{}.pkl')
    train_path = data_path.format(args.dataset_name, 'train', args.num_frames, args.num_pairs)
    val_path = data_path.format(args.dataset_name, 'val', args.num_frames, args.num_pairs)
    test_path = data_path.format(args.dataset_name, 'test', args.num_frames, args.num_pairs)

    if not args.force and os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path):
        logging.info('Found all data sets on disk.')
        dataloaders = []
        for data_type in ['train', 'val', 'test']:
            logging.info('Loading {} data set...'.format(data_type))
            dataset = load_object(data_path.format(args.dataset_name, data_type, \
                                                   args.num_frames, args.num_pairs))
            shuffle = 1 if data_type == 'train' else False
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=4)
            dataloaders.append(dataloader)
            logging.info('Done.')

    else:
        logging.info('Did not find at least one data set on disk. Creating all 3...')

        data_dict = {}
        for data_type in ['train', 'val', 'test']:
            data_dict[data_type] = load_data(args, data_type)

        sequence_length = data_dict['train'].shape[1]
        max_frame_diff = np.hstack([bucket for bucket in args.time_buckets]).max()
        assert max_frame_diff <= sequence_length-args.num_frames, \
            'Cannot have difference of {} when sequence length is {} and number of \
            stacked frames are {}'.format(max_frame_diff, sequence_length, args.num_frames)

        # Normalize data
        transforms, mean, std = get_normalize_transform(args, data_dict['train'])
        imshow(data_dict['train'], args)

        time_buckets_dict = get_time_buckets_dict(args)
        differences_dict = get_candidates_differences_dict(args, sequence_length, max_frame_diff)

        dataloaders = []
        for data_type in ['train', 'val', 'test']:
            logging.info('Generating {} data set...'.format(data_type))
            stacked_img_pairs, target_buckets = np.array([]), np.array([])
            target_differences, target_frame_numbers = np.array([]), np.array([])

            for difference in range(max_frame_diff+1):
                img_pairs, buckets, differences, frame_numbers = \
                    get_samples_at_difference(args, data_dict[data_type], difference, \
                                              differences_dict, time_buckets_dict)
                stacked_img_pairs = np.vstack((stacked_img_pairs, img_pairs)) if stacked_img_pairs.size \
                                                                              else img_pairs
                target_buckets = np.append(target_buckets, buckets)
                target_differences = np.append(target_differences, differences)
                target_frame_numbers = np.vstack((target_frame_numbers, frame_numbers)) \
                                        if target_frame_numbers.size else frame_numbers

            dataset = OfflineMovingMNISTDataset(stacked_img_pairs, target_buckets, target_differences, \
                                                target_frame_numbers, transforms=transforms)
            logging.info('Done. Dumping {} data set to disk...'.format(data_type))
            save_object(dataset, data_path.format(args.dataset_name, data_type, args.num_frames, \
                                                  args.num_pairs))
            logging.info('Done.')

            shuffle = True if data_type == 'train' else False
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=4)
            dataloaders.append(dataloader)

    return dataloaders

def load_data(args, data_type):
    filename_dataset = '{}_{}{}'.format(args.dataset_name, data_type, args.data_ext)
    logging.info('Loading "{}"...'.format(filename_dataset))
    file_path = os.path.join(args.project_dir, args.data_dir, filename_dataset)

    if args.data_ext == '.npy':
        data = np.load(file_path)
    elif args.data_ext == '.pkl':
        data = load_object(file_path)
    elif args.data_ext == '.h5':
        with h5py.File(file_path) as f:
            data = np.array(f['inputs'])
    else:
        raise ValueError('Unknown file extension "{}"!'.format(args.data_ext))
    logging.info('Done.')

    return data

def load_maze_data(args, data_type):
    filename_dataset = '{}{}.h5'.format(args.dataset_name, '_{}'.format(data_type) if data_type else '')
    file_path = os.path.join(args.project_dir, args.data_dir, filename_dataset)
    if not args.force and os.path.exists(file_path):
        logging.info('Loading "{}"...'.format(file_path))
        with h5py.File(file_path, 'r') as f:
            keys = sorted([int(key) for key in f.keys()])
            data = [f[str(key)][:] for key in keys]
        logging.info('Done.')
    else:
        data = split_and_dump_maze_data(args, data_type)

    seq_lens = np.array([maze.shape[0] for maze in data])
    logging.info('Min/Max/Avg/Total sequence length in {} data: '\
                 '{:.0f}/{:.0f}/{:.0f}/{:.0f}'.format(data_type, \
                 np.min(seq_lens), np.max(seq_lens), np.mean(seq_lens), np.sum(seq_lens)))
    if args.flatten:
        data = np.vstack([maze for maze in data])[np.newaxis,:]
    return data

def split_and_dump_maze_data(args, data_type, val_size=0.2, test_size=0.2):
    file_path_in = os.path.join(args.project_dir, args.data_dir, '{}.h5'.format(args.dataset_name))
    logging.info('Loading "{}"...'.format(file_path_in))
    with h5py.File(file_path_in, 'r') as f_in:
        all_data = []
        keys = sorted([int(key) for key in f_in.keys()])
        all_data = [f_in[str(key)][:] for key in keys]
    logging.info('Done.')

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

    file_path_out = os.path.join(args.project_dir, args.data_dir, '{}_{}.h5'.format(args.dataset_name, data_type))
    logging.info('Dumping and returning {}...'.format(file_path_out))
    with h5py.File(file_path_out, 'w') as f_out:
        for i in range(len(data_dict[data_type])):
            # Retain the original keys when splitting into train/val/test
            f_out.create_dataset(str(start_idx[data_type]+i), data=data_dict[data_type][i])
    logging.info('Done.')

    return data_dict[data_type]

def get_normalize_transform(args, data):
    # Calculate mean, std from train data, and normalize
    mean = np.mean(data)
    std = np.std(data)

    normalize = transforms.Compose([
        transforms.Normalize((mean,)*args.num_frames, (std,)*args.num_frames)
    ])

    return normalize, mean, std

def get_time_buckets_dict(args):
    '''
    Returns a dict, with the time diff
    as its key and the target class (0-indexed) for it as its value
    '''
    logging.info('Getting time buckets dictionary...')
    bucket_idx = 0
    buckets_dict = {}
    for bucket in args.time_buckets:
        for time in bucket:
            buckets_dict[time] = bucket_idx
        bucket_idx += 1
    logging.info('Done.')
    return buckets_dict

def get_candidates_differences_dict(args, sequence_length, max_frame_diff):
    '''
    Returns a dict with the key as the time difference between the frames
    and the value as a list of tuples (start_frame, end_frame) containing
    all the pair of frames with that diff in time
    '''
    logging.info('Getting frame differences dictionary...')
    differences_dict = {}
    differences = range(max_frame_diff+1)
    for diff in differences:
        start_frame = args.num_frames-1
        end_frame = start_frame+diff
        while end_frame <= sequence_length-1:
            differences_dict.setdefault(diff, []).append(tuple((start_frame, end_frame)))
            start_frame += 1
            end_frame += 1
    logging.info('Done.')
    return differences_dict

def get_samples_at_difference(args, data, difference, differences_dict, time_buckets_dict):
    '''
    Returns samples by selecting the list of tuples for the associated time difference,
    sampling the num of pairs per video example, and finally returning the (stacked) image pairs,
    their associated buckets (classes), frame difference, and last frame numbers for each pair (tuple)
    '''
    logging.info('Getting all pairs with a frame difference of {}...'.format(difference))
    img_pairs, target_buckets, differences, frame_numbers = [], [], [], []
    candidates = differences_dict[difference]
    idx_pairs = np.random.choice(len(candidates), size=args.num_pairs)
    for row in data:
        for idx_pair in idx_pairs:
            target1_last_frame, target2_last_frame = candidates[idx_pair]
            target1_frames = range(target1_last_frame-args.num_frames+1, target1_last_frame+1)
            target2_frames = range(target2_last_frame-args.num_frames+1, target2_last_frame+1)
            img_pairs.append([row[target1_frames], row[target2_frames]])
            bucket = time_buckets_dict[difference]
            target_buckets.append(bucket)
            differences.append(difference)
            frame_numbers.append(tuple((target1_last_frame, target2_last_frame)))
    logging.info('Done.')
    return np.array(img_pairs), np.array(target_buckets), np.array(differences), np.array(frame_numbers)

def split_data(data, val_size, test_size):
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

def generate_embedding_dataloader(args):
    assert args.dataset_type in ['maze'], \
        'Unknown dataset type "{}" passed.'.format(args.dataset_type)

    logging.info('Generating data loader...')

    # Maze Dataset
    data = load_maze_data(args, '')
    assert len(data[0].shape) == 4, 'Unknown input data shape "{}"'.format(data.shape)
    assert args.emb_model in ['cnn', 'emb-cnn1', 'emb-cnn2', 'rel'], \
        'Unknown embedding network model name "{}" passed.'.format(args.emb_model)

    dataset = MazeEmbeddingsDataset(data, args.num_channels, return_indices=False \
                                    if args.emb_model == 'cnn' else True)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    logging.info('Done.')

    return dataloader
