import numpy as np
import pandas as pd
import os
import pickle
import torch
from torchvision.utils import make_grid
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, random_split

from capstone_project.utils import save_plot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load_data(project_dir, data_dir, filename):
    filename = os.path.join(project_dir, data_dir, filename)
    data = np.load(filename)
    data = data.swapaxes(0,1)
    return data

def get_time_buckets_dict(time_buckets):
    bucket_idx = 0
    buckets_dict = {}
    for bucket in time_buckets:
        for time in bucket:
            buckets_dict[time] = bucket_idx
        bucket_idx += 1
    return buckets_dict

def get_frame_differences_dict(num_total_frames):
    # min_diff = 0, max_diff = num_total_frames-1
    # TODO: NUM_FRAMES_IN_STACK
    differences = range(num_total_frames)
    differences_dict = {}
    for diff in differences:
        for i in range(num_total_frames):
            if i+diff >= num_total_frames:
                break
            start_frame, end_frame = i, i+diff
            while end_frame < num_total_frames:
                differences_dict.setdefault(diff, []).append((start_frame, end_frame))
                end_frame += 1
    return differences_dict

def get_samples_at_difference(data, difference, differences_dict, num_videos_per_row, time_buckets_dict):
    video_pairs, y = [], []
    candidates = differences_dict[difference]
    np.random.seed(1337)
    idx_pairs = np.random.choice(len(candidates), size=num_videos_per_row)
    for row in data:
        for idx_pair in idx_pairs:
            target1, target2 = candidates[idx_pair]
            video_pairs.append([row[target1], row[target2]])
            bucket = time_buckets_dict[difference]
            y.append(bucket)
    return np.array(video_pairs), np.array(y)

def get_paired_data(project_dir, data_dir, plots_dir, filename, time_buckets, num_rows=2, num_videos_per_row=1, force=False):
    mean, std = 0, 1
    data = load_data(project_dir, data_dir, filename)
    imshow(data, mean, std, project_dir, plots_dir)

    X_path = os.path.join(project_dir, data_dir, 'X.pkl')
    y_path = os.path.join(project_dir, data_dir, 'y.pkl')
    if not force and os.path.exists(X_path) and os.path.exists(y_path):
        data = None
        X = pickle.load(open(X_path, 'rb'))
        y = pickle.load(open(y_path, 'rb'))
        return X, y
    num_total_frames = data.shape[1]
    time_buckets_dict = get_time_buckets_dict(time_buckets)
    differences_dict = get_frame_differences_dict(num_total_frames)
    X, y = np.array([]), np.array([])
    for i in range(num_rows):
        for difference in range(num_total_frames):
            video_pairs, targets = get_samples_at_difference(data, difference, differences_dict, num_videos_per_row, time_buckets_dict)
            X = video_pairs if not X.size else np.vstack((X, video_pairs))
            y = np.append(y, targets)
    pickle.dump(X, open(X_path, 'wb'))
    pickle.dump(y, open(y_path, 'wb'))
    return X, y

class MovingMNISTDataset(Dataset):
    def __init__(self, X, y, transforms=None):
        self.X = X
        self.y = y
        self.transforms = transforms

    def __getitem__(self, index):
        # Load data and get label
        X = self.X[index]
        y = self.y[index]
        return X, y

    def __len__(self):
        return len(self.y)

def generate_dataloader(X, y, test_size, val_size, batch_size, project_dir, plots_dir):
    # mean = np.mean(dataset)
    # std = np.std(dataset)
    # dataset = (dataset - mean)/std
    # data_transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((mean,), (std,))
    # ])

    X, y = torch.from_numpy(X), torch.from_numpy(y)
    dataset = dataset = MovingMNISTDataset(X, y, transforms=None)

    num_test = int(np.floor(test_size*len(dataset)))
    num_train_val = len(dataset) - num_test
    num_val = int(np.floor(num_train_val*val_size/(1 - test_size)))
    num_train = num_train_val - num_val

    train_dataset, val_dataset, test_dataset = random_split(dataset, [num_train, num_val, num_test])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader

def imshow(data, mean, std, project_dir, plots_dir):
    image_dim = data.shape[-1]
    np.random.seed(1337)
    images = data[np.random.choice(len(data), size=1)]
    images = torch.from_numpy(images)

    images = make_grid(images[0].reshape(-1, 1, image_dim, image_dim), nrow=10, padding=5, pad_value=1)
    images = images*std + mean  # unnormalize
    np_image = images.numpy()

    fig = plt.figure(figsize=(30, 10))
    plt.imshow(np.transpose(np_image, axes=(1, 2, 0)))
    plt.tight_layout()
    save_plot(project_dir, plots_dir, fig, 'data_sample.png')
