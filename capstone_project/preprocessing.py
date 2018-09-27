import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data.sampler import Sampler

from capstone_project.utils import save_plot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def imshow(data_loader, project_dir):
    data_iter = iter(data_loader)
    images = data_iter.next()

    images = make_grid(images[0].reshape(-1, 1, 64, 64), nrow=10)
    np_image = images.numpy()

    fig = plt.figure(figsize=(50, 20))
    plt.imshow(np.transpose(np_image, axes=(1, 2, 0)))
    save_plot(project_dir, fig, 'data_sample.png')

def generate_dataloader(dataset, test_size, val_size, batch_size):
    dataset = torch.from_numpy(dataset)

    num_test = int(np.floor(test_size*len(dataset)))
    num_train_val = len(dataset) - num_test
    num_val = int(np.floor(num_train_val*val_size/(1 - test_size)))
    num_train = num_train_val - num_val

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [num_train, num_val, num_test])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    imshow(train_loader)

    return train_loader, val_loader, test_loader
