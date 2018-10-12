import numpy as np
import pandas as pd
import os
import pickle
import torch
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, random_split

from capstone_project.utils import imshow

def load_data(project_dir, data_dir, filename):
    print('Loading "{}"... '.format(filename), end='', flush=True)
    filename = os.path.join(project_dir, data_dir, filename)
    data = np.load(filename)
    data = data.swapaxes(0,1)
    print('Done.')
    return data

# This function returns a dict, with the time diff
# as it's key and the class for it as it's value
def get_time_buckets_dict(time_buckets):
    '''
    Returns a dict, with the time diff
    as its key and the target class (0-indexed) for it as its value
    '''
    print('Getting time buckets dictionary... ', end='', flush=True)
    bucket_idx = 0
    buckets_dict = {}
    for bucket in time_buckets:
        for time in bucket:
            buckets_dict[time] = bucket_idx
        bucket_idx += 1
    print('Done.')
    return buckets_dict

def get_frame_differences_dict(max_frame, num_frames_in_stack):
    '''
    Returns a dict with the key as the time difference between the frames
    and the value as a list of tuples (start_frame, end_frame) containing
    all the pair of frames with that diff in time
    '''
    print('Getting frame differences dictionary... ', end='', flush=True)
    differences = range(max_frame-num_frames_in_stack)
    differences_dict = {}
    for diff in differences:
        i = num_frames_in_stack-1
        if i+diff > max_frame:
            break
        last_start_frame, last_end_frame = i, i+diff
        while last_end_frame <= max_frame:
            differences_dict.setdefault(diff, []).append(tuple((last_start_frame, last_end_frame)))
            last_start_frame += 1
            last_end_frame += 1
    print('Done.')
    return differences_dict

def get_samples_at_difference(data, difference, differences_dict, num_pairs_per_example, num_frames_in_stack, time_buckets_dict):
    '''
    The task of this function is to get the samples by first selecting the list of tuples
    for the associated time difference, and then sampling the num of pairs per video example
    and then finally returning, the video pairs and their associated class(for the time bucket)
    '''
    print('Getting all pairs with a frame difference of {}... '.format(difference), end='', flush=True)
    video_pairs, y = [], []
    candidates = differences_dict[difference]
    np.random.seed(1337)
    idx_pairs = np.random.choice(len(candidates), size=num_pairs_per_example)
    for row in data:
        for idx_pair in idx_pairs:
            target1_last_frame, target2_last_frame = candidates[idx_pair]
            target1_frames = list(range(target1_last_frame-num_frames_in_stack+1, target1_last_frame+1))
            target2_frames = list(range(target2_last_frame-num_frames_in_stack+1, target2_last_frame+1))
            video_pairs.append([row[target1_frames], row[target2_frames]])
            bucket = time_buckets_dict[difference]
            y.append(bucket)
    print('Done.')
    return np.array(video_pairs), np.array(y)

def get_paired_data(project_dir, data_dir, plots_dir, filename, time_buckets, num_passes_for_generation=2, num_pairs_per_example=1, num_frames_in_stack=2, force=False):
    mean, std = 0, 1
    data = load_data(project_dir, data_dir, filename)
    imshow(data, mean, std, project_dir, plots_dir)

    filename = '.'.join(filename.split('.')[:-1])
    X_path = os.path.join(project_dir, data_dir, '{}_X.pkl'.format(filename))
    y_path = os.path.join(project_dir, data_dir, '{}_y.pkl'.format(filename))
    if not force and os.path.exists(X_path) and os.path.exists(y_path):
        data = None
        print('Found existing data. Loading it... ', end='', flush=True)
        X = pickle.load(open(X_path, 'rb'))
        y = pickle.load(open(y_path, 'rb'))
        print('Done.')
        return X, y
    print('Did not find existing data. Creating it... ')
    time_buckets_dict = get_time_buckets_dict(time_buckets)
    max_frame = np.hstack([bucket for bucket in time_buckets]).max()
    differences_dict = get_frame_differences_dict(max_frame, num_frames_in_stack)
    X, y = np.array([]), np.array([])
    for i in range(num_passes_for_generation):
        print('Making pass {} through data... '.format(i+1))
        for difference in range(max_frame-num_frames_in_stack):
            video_pairs, targets = get_samples_at_difference(data, difference, differences_dict, num_pairs_per_example, num_frames_in_stack, time_buckets_dict)
            X = np.vstack((X, video_pairs)) if X.size else video_pairs
            y = np.append(y, targets)
        print('Done.')
    print('Data generation done. Dumping data to disk... ', end='', flush=True)
    pickle.dump(X, open(X_path, 'wb'), protocol=4)
    pickle.dump(y, open(y_path, 'wb'))
    print('Done.')
    return X, y

class MovingMNISTDataset(Dataset):
    # TODO: write function for implementing transforms
    def __init__(self, X, y, transforms=None):
        self.X = X
        self.y = y
        self.transforms = transforms

    def __getitem__(self, index):
        # Load data and get label
        x1 = self.X[index][0]
        x2 = self.X[index][1]
        y = self.y[index]
        return x1, x2, y

    def __len__(self):
        return len(self.y)

def generate_dataloader(X, y, test_size, val_size, batch_size, project_dir, plots_dir):
    # TODO: Normalize data
    # mean = np.mean(dataset)
    # std = np.std(dataset)
    # dataset = (dataset - mean)/std
    # data_transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((mean,), (std,))
    # ])
    print('Generating train, val, and test data loaders... ', end='', flush=True)

    X, y = torch.from_numpy(X), torch.from_numpy(y)
    dataset = MovingMNISTDataset(X, y, transforms=None)

    num_test = int(np.floor(test_size*len(dataset)))
    num_train_val = len(dataset) - num_test
    num_val = int(np.floor(num_train_val*val_size/(1 - test_size)))
    num_train = num_train_val - num_val

    train_dataset, val_dataset, test_dataset = random_split(dataset, [num_train, num_val, num_test])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    print('Done.')

    return train_loader, val_loader, test_loader
