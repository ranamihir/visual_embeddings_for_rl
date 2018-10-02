import numpy as np
import pandas as pd
import os
import torch
from torch.autograd import Variable
from torchvision.utils import make_grid

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def train(embedding_network, classification_network, dataloader, criterion, optimizer, device, epoch):
    embedding_network.train()
    classification_network.train()
    loss_train = 0.
    for batch_idx, (x1, x2, y) in enumerate(dataloader):
        x1, x2, y = Variable(x1).to(device).float(), Variable(x2).to(device).float(), Variable(y).to(device).long()
        optimizer.zero_grad()
        embedding_output1 = embedding_network(x1)
        embedding_output2 = embedding_network(x2)
        classification_output = classification_network(embedding_output1, embedding_output2)
        loss = criterion(classification_output, y)
        loss.backward()
        optimizer.step()

        # Accurately compute loss, because of different batch size
        loss_train += loss.item() * len(x1) / len(dataloader.dataset)

        if batch_idx % (len(dataloader.dataset)//(5*y.shape[0])) == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * y.shape[0], len(dataloader.dataset),
                100. * batch_idx / len(dataloader), loss.item()))

    optimizer.zero_grad()
    return loss_train

def test(embedding_network, classification_network, dataloader, criterion, device):
    embedding_network.eval()
    classification_network.eval()
    loss_test = 0.
    y_ls = []
    output_ls = []
    with torch.no_grad():
        for batch_idx, (x1, x2, y) in enumerate(dataloader):
            x1, x2, y = Variable(x1).to(device).float(), Variable(x2).to(device).float(), Variable(y).to(device).long()
            embedding_output1 = embedding_network(x1)
            embedding_output2 = embedding_network(x2)
            classification_output = classification_network(embedding_output1, embedding_output2)
            loss = criterion(classification_output, y)

            # Accurately compute loss, because of different batch size
            loss_test += loss.item() / len(dataloader.dataset)

            output_ls.append(classification_output)
            y_ls.append(y)
    return loss_test, torch.cat(output_ls, dim=0), torch.cat(y_ls, dim=0)

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

def imshow(data, mean, std, project_dir, plots_dir):
    print('Plotting sample data and saving to "data_sample.png"... ', end='', flush=True)
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
    print('Done.')

def save_plot(project_dir, plots_dir, fig, filename):
    plot_path = os.path.join(project_dir, plots_dir)
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    fig.savefig(os.path.join(plot_path, filename))

def make_dirs(project_dir, directories_to_create):
    for directory in directories_to_create:
        directory_path = os.path.join(project_dir, directory)
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

def save_checkpoint(embedding_network, classification_network, optimizer, train_loss_history, \
                    val_loss_history, train_accuracy_history, val_accuracy_history, epoch, checkpoints_dir='checkpoints'):
    embedding_state = {'state_dict': embedding_network.state_dict()}
    classification_state = {'state_dict': classification_network.state_dict(), 'optimizer': optimizer.state_dict(), \
        'epoch': epoch, 'train_loss_history': train_loss_history, 'train_loss_history': val_loss_history, \
        'train_accuracy_history': train_accuracy_history, 'train_accuracy_history': val_accuracy_history}
    torch.save(embedding_state, os.path.join(checkpoints_dir, \
        'embedding_network_{}.pkl'.format(epoch)))
    torch.save(classification_state, os.path.join(checkpoints_dir, \
        'classification_network_{}.pkl'.format(epoch)))

def load_checkpoint(embedding_network, classification_network, optimizer, device, epoch, checkpoints_dir='checkpoints'):
    train_loss_history, val_loss_history = [], []
    train_accuracy_history, val_accuracy_history = [], []

    # Note: Input model & optimizer should be pre-defined. This routine only updates their states.
    embedding_path = os.path.join(checkpoints_dir, 'embedding_network_{}.pkl'.format(epoch))
    classification_path = os.path.join(checkpoints_dir, 'classification_network_{}.pkl'.format(epoch))
    if os.path.isfile(embedding_path) and os.path.isfile(classification_path):
        print('Loading checkpoint "{}" and "{}"...'.format(embedding_path, classification_path), end='', flush=True)
        embedding_checkpoint = torch.load(embedding_path)
        classification_checkpoint = torch.load(classification_path)
        assert epoch == classification_checkpoint['epoch']
        embedding_network.load_state_dict(embedding_checkpoint['state_dict'])
        classification_network.load_state_dict(classification_checkpoint['state_dict'])
        optimizer.load_state_dict(classification_checkpoint['optimizer'])
        train_loss_history = classification_checkpoint['train_loss_history']
        val_loss_history = classification_checkpoint['val_loss_history']
        train_accuracy_history = classification_checkpoint['train_accuracy_history']
        val_accuracy_history = classification_checkpoint['val_accuracy_history']

        embedding_network = embedding_network.to(device)
        classification_network = classification_network.to(device)
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

        print('Loaded checkpoint "{}".'.format(embedding_path, classification_path))

    else:
        print('No checkpoint found at "{}" and/or "{}"!'.format(embedding_path, classification_path))

    return embedding_network, classification_network, optimizer, train_loss_history, val_loss_history, train_accuracy_history, val_accuracy_history
