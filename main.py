import numpy as np
import pandas as pd
import os
import argparse
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data.sampler import Sampler

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchviz import make_dot, make_dot_from_trace

from capstone_project.preprocessing import get_paired_data, generate_dataloader
from capstone_project.models.embedding_network import EmbeddingNetwork
from capstone_project.models.classification_network import ClassificationNetwork
from capstone_project.utils import train, test, accuracy, save_plot, make_dirs, save_checkpoint, load_checkpoint


parser = argparse.ArgumentParser()
parser.add_argument('--project-dir', metavar='PROJECT_DIR', dest='project_dir', help='path to project directory', required=False)
parser.add_argument('--dataset', metavar='DATASET', dest='dataset', help='name of dataset file in data directory', required=False)
parser.add_argument('--data-dir', metavar='DATA_DIR', dest='data_dir', help='path to data directory', required=False, default='data')
parser.add_argument('--batch-size', metavar='BATCH_SIZE', dest='batch_size', help='batch size', required=False, type=int, default=128)
parser.add_argument('--epochs', metavar='EPOCHS', dest='epochs', help='number of epochs', required=False, type=int, default=10)
parser.add_argument('--device', metavar='DEVICE', dest='device', help='device', required=False)
parser.add_argument('--device-id', metavar='DEVICE_ID', dest='device_id', help='device id of gpu', required=False, type=int)
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--lr', metavar='LR', dest='lr', help='learning rate', required=False, type=float, default=1e-4)
parser.add_argument('--force', action='store_true', help='overwrites all existing data')
args = parser.parse_args()


# Globals
PROJECT_DIR = args.project_dir if args.project_dir else '/home/mihir/Desktop/GitHub/nyu/capstone_project/'
DATA_DIR, CHECKPOINTS_DIR, PLOTS_DIR = args.data_dir, 'checkpoints', 'plots'
DATASET = args.dataset if args.dataset else 'mnist_test_seq.npy'
TEST_SIZE, VAL_SIZE = 0.2, 0.2

BATCH_SIZE = args.batch_size    # input batch size for training
N_EPOCHS = args.epochs          # number of epochs to train
LR = args.lr                    # learning rate
NGPU = args.ngpu                # number of GPUs
DEVICE = args.device if args.device else 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if args.device_id and DEVICE == 'cuda':
    torch.cuda.set_device(args.device_id)

NUM_PASSES_FOR_GENERATION = 1   # number of passes through data for pair generation
NUM_FRAMES_IN_STACK = 2         # number of (total) frames to concatenate for each video
NUM_PAIRS_PER_EXAMPLE = 5       # number of pairs to generate for given video and time difference
TIME_BUCKETS = [[0], [1], [2], [3,4], list(range(5,11,1)), list(range(11,20-NUM_FRAMES_IN_STACK+1,1))]

def main():
    # Create directories for storing checkpoints and plots if not present
    make_dirs(PROJECT_DIR, [CHECKPOINTS_DIR, PLOTS_DIR])

    # TODO: make graphviz work

    X, y = get_paired_data(PROJECT_DIR, DATA_DIR, PLOTS_DIR, DATASET, TIME_BUCKETS, NUM_PASSES_FOR_GENERATION, NUM_PAIRS_PER_EXAMPLE, NUM_FRAMES_IN_STACK, force=args.force)
    train_loader, val_loader, test_loader = generate_dataloader(X, y, TEST_SIZE, VAL_SIZE, BATCH_SIZE, PROJECT_DIR, PLOTS_DIR)

    # Network hyperparameters
    in_dim, in_channels, out_dim = 64, NUM_FRAMES_IN_STACK, 1024
    embedding_hidden_size, classification_hidden_size = 1024, 1024
    num_outputs = 6

    print('Creating models... ', end='', flush=True)
    embedding_network = EmbeddingNetwork(in_dim, in_channels, out_dim, embedding_hidden_size).to(DEVICE)
    classification_network = ClassificationNetwork(out_dim, num_outputs, classification_hidden_size).to(DEVICE)
    print('Done.')
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     embedding_network = nn.DataParallel(embedding_network)
    #     embedding_network.to(device)
    #     classification_network = nn.DataParallel(classification_network)
    #     classification_network.to(device)
    criterion_train = nn.CrossEntropyLoss()
    criterion_test = nn.CrossEntropyLoss(reduction='sum')
    optimizer = optim.Adam(list(embedding_network.parameters()) + list(classification_network.parameters()), lr=LR)

    train_loss_history, train_accuracy_history = [], []
    val_loss_history, val_accuracy_history = [], []
    stop_epoch = N_EPOCHS

    # Uncomment this line to load model already dumped
    # embedding_network, classification_network, optimizer, train_loss_history, val_loss_history = \
        # load_checkpoint(embedding_network, classification_network, optimizer, DEVICE, 1, CHECKPOINTS_DIR)

    for epoch in range(1, N_EPOCHS+1):
        try:
            train_loss = train(
                embedding_network=embedding_network,
                classification_network=classification_network,
                criterion=criterion_train,
                dataloader=train_loader,
                optimizer=optimizer,
                device=DEVICE,
                epoch=epoch
            )

            val_loss, val_pred, val_true = test(
                embedding_network=embedding_network,
                classification_network=classification_network,
                dataloader=val_loader,
                criterion=criterion_test,
                device=DEVICE
            )

            accuracy_train = accuracy(embedding_network, classification_network, train_loader, criterion_test, DEVICE)
            accuracy_val = accuracy(embedding_network, classification_network, val_loader, criterion_test, DEVICE)
            train_loss_history.append(train_loss)
            val_loss_history.append(val_loss)
            train_accuracy_history.append(accuracy_train)
            val_accuracy_history.append(accuracy_val)

            print('TRAIN Epoch: {}\tAverage loss: {:.4f}, Accuracy: {:.0f}%'.format(epoch, train_loss, accuracy_train))
            print('VAL   Epoch: {}\tAverage loss: {:.4f}, Accuracy: {:.0f}%\n'.format(epoch, val_loss, accuracy_val))
        except KeyboardInterrupt:
            # Save the model checkpoints
            print('Keyboard Interrupted!')
            stop_epoch = epoch
            break

    # Save the model checkpoint
    print('Dumping model and results... ', end='', flush=True)
    save_checkpoint(embedding_network, classification_network, optimizer, train_loss_history, val_loss_history, \
        train_accuracy_history, val_accuracy_history, stop_epoch, CHECKPOINTS_DIR)
    print('Done.')

    print('Saving and plotting loss and accuracy histories... ', end='', flush=True)
    fig = plt.figure()
    loss_history_df = pd.DataFrame({
        'train': train_loss_history,
        'test': val_loss_history,
    })
    loss_history_df.plot(alpha=0.5, figsize=(10,8))
    save_plot(PROJECT_DIR, PLOTS_DIR, fig, 'loss_vs_iterations.png')

    fig = plt.figure()
    accuracy_history_df = pd.DataFrame({
        'train': train_accuracy_history,
        'test': val_accuracy_history,
    })
    accuracy_history_df.plot(alpha=0.5, figsize=(10,8))
    save_plot(PROJECT_DIR, PLOTS_DIR, fig, 'accuracies_vs_iterations.png')
    print('Done.')

    print('Generating and saving visualization of computational graph... ', end='', flush=True)
    with torch.onnx.set_training(embedding_network, False) and torch.onnx.set_training(classification_network, False):
        embedding_output1 = embedding_network(train_loader.dataset[0][0].view(-1, 1, 64, 64).to(DEVICE).float())
        embedding_output2 = embedding_network(train_loader.dataset[1][0].view(-1, 1, 64, 64).to(DEVICE).float())
        trace, _ = torch.jit.get_trace_graph(classification_network, args=(embedding_output1, embedding_output2,))
        with open(os.path.join(PROJECT_DIR, PLOTS_DIR, 'model_DAG.svg'), 'w') as f:
            f.write(make_dot_from_trace(trace)._repr_svg_())
    print('Done.')

if __name__ == '__main__':
    main()
