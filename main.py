import numpy as np
import pandas as pd
import os
import argparse
import pickle
import logging

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
from capstone_project.utils import *


parser = argparse.ArgumentParser()
parser.add_argument('--project-dir', metavar='PROJECT_DIR', dest='project_dir', help='path to project directory', required=False)
parser.add_argument('--dataset', metavar='DATASET', dest='dataset', help='name of dataset file in data directory', required=False)
parser.add_argument('--data-dir', metavar='DATA_DIR', dest='data_dir', help='path to data directory (used if different from "data")', \
                    required=False, default='data')
parser.add_argument('--batch-size', metavar='BATCH_SIZE', dest='batch_size', help='batch size', required=False, type=int, default=64)
parser.add_argument('--epochs', metavar='EPOCHS', dest='epochs', help='number of epochs', required=False, type=int, default=10)
parser.add_argument('--device', metavar='DEVICE', dest='device', help='device', required=False)
parser.add_argument('--device-id', metavar='DEVICE_ID', dest='device_id', help='device id of gpu', required=False, type=int)
parser.add_argument('--ngpu', metavar='NGPU', dest='ngpu', help='number of GPUs to use', required=False, type=int, default=1)
parser.add_argument('--lr', metavar='LR', dest='lr', help='learning rate', required=False, type=float, default=1e-4)
parser.add_argument('--force', action='store_true', help='overwrites all existing data')
# TODO: load-ckpt
parser.add_argument('--checkpoints-dir', metavar='CHECKPOINTS_DIR', dest='checkpoints_dir', help='path to checkpoints directory', required=False)
args = parser.parse_args()


# Globals
PROJECT_DIR = args.project_dir if args.project_dir else '/home/mihir/Desktop/GitHub/nyu/learning_visual_embeddings/'
DATA_DIR,  PLOTS_DIR, LOGGING_DIR = args.data_dir, 'plots', 'logs'
CHECKPOINTS_DIR = args.checkpoints_dir if args.checkpoints_dir else 'checkpoints'
DATASET = args.dataset if args.dataset else 'mnist_test_seq.npy'
TEST_SIZE, VAL_SIZE = 0.2, 0.2

BATCH_SIZE = args.batch_size    # input batch size for training
N_EPOCHS = args.epochs          # number of epochs to train
LR = args.lr                    # learning rate
NGPU = args.ngpu                # number of GPUs
# TODO: Incorporate NGPUs
DEVICE = args.device if args.device else 'cuda' if torch.cuda.is_available() else 'cpu'
if args.device_id and DEVICE == 'cuda':
    DEVICE_ID = args.device_id
    torch.cuda.set_device(DEVICE_ID)

NUM_PASSES_FOR_GENERATION = 1   # number of passes through data for pair generation
NUM_FRAMES_IN_STACK = 2         # number of (total) frames to concatenate for each video
NUM_PAIRS_PER_EXAMPLE = 5       # number of pairs to generate for given video and time difference
TIME_BUCKETS = [[0], [1], [2], [3,4], range(5,11,1)]

def main():
    make_dirs(PROJECT_DIR, [CHECKPOINTS_DIR, PLOTS_DIR, LOGGING_DIR]) # Create all required directories if not present
    setup_logging(PROJECT_DIR, LOGGING_DIR) # Setup configuration for logging
    print_config(globals().copy()) # Print all global variables defined above

    # TODO: make graphviz work

    X, y = get_paired_data(PROJECT_DIR, DATA_DIR, PLOTS_DIR, DATASET, TIME_BUCKETS, NUM_PASSES_FOR_GENERATION, NUM_PAIRS_PER_EXAMPLE, NUM_FRAMES_IN_STACK, force=args.force)
    train_loader, val_loader, test_loader = generate_dataloader(X, y, TEST_SIZE, VAL_SIZE, BATCH_SIZE, PROJECT_DIR, PLOTS_DIR)

    # Network hyperparameters
    in_dim, in_channels, out_dim = X.shape[-1], NUM_FRAMES_IN_STACK, 1024
    embedding_hidden_size, classification_hidden_size = 1024, 1024
    num_outputs = len(TIME_BUCKETS)

    logging.info('Creating models...')
    embedding_network = EmbeddingNetwork(in_dim, in_channels, embedding_hidden_size, out_dim).to(DEVICE)
    classification_network = ClassificationNetwork(out_dim, classification_hidden_size, num_outputs).to(DEVICE)
    logging.info('Done.')
    # if torch.cuda.device_count() > 1:
    #     logging.info("Let's use", torch.cuda.device_count(), "GPUs!")
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

            logging.info('TRAIN Epoch: {}\tAverage loss: {:.4f}, Accuracy: {:.0f}%'.format(epoch, train_loss, accuracy_train))
            logging.info('VAL   Epoch: {}\tAverage loss: {:.4f}, Accuracy: {:.0f}%\n'.format(epoch, val_loss, accuracy_val))
        except KeyboardInterrupt:
            # Save the model checkpoints
            logging.info('Keyboard Interrupted!')
            stop_epoch = epoch
            break

    # Save the model checkpoint
    logging.info('Dumping model and results...')
    save_checkpoint(embedding_network, classification_network, optimizer, train_loss_history, val_loss_history, \
        train_accuracy_history, val_accuracy_history, stop_epoch, CHECKPOINTS_DIR)
    logging.info('Done.')

    logging.info('Saving and plotting loss and accuracy histories...')
    fig = plt.figure()
    loss_history_df = pd.DataFrame({
        'train': train_loss_history,
        'test': val_loss_history,
    })
    loss_history_df.plot(alpha=0.5, figsize=(10, 8))
    save_plot(PROJECT_DIR, PLOTS_DIR, fig, 'loss_vs_iterations.png')

    fig = plt.figure()
    accuracy_history_df = pd.DataFrame({
        'train': train_accuracy_history,
        'test': val_accuracy_history,
    })
    accuracy_history_df.plot(alpha=0.5, figsize=(10, 8))
    save_plot(PROJECT_DIR, PLOTS_DIR, fig, 'accuracies_vs_iterations.png')
    logging.info('Done.')

    logging.info('Generating and saving visualization of computational graph...')
    with torch.onnx.set_training(embedding_network, False) and torch.onnx.set_training(classification_network, False):
        embedding_output1 = embedding_network(train_loader.dataset[0][0].view(-1, 1, in_dim, in_dim).to(DEVICE).float())
        embedding_output2 = embedding_network(train_loader.dataset[1][0].view(-1, 1, in_dim, in_dim).to(DEVICE).float())
        trace, _ = torch.jit.get_trace_graph(classification_network, args=(embedding_output1, embedding_output2,))
        with open(os.path.join(PROJECT_DIR, PLOTS_DIR, 'model_DAG.svg'), 'w') as f:
            f.write(make_dot_from_trace(trace)._repr_svg_())
    logging.info('Done.')

if __name__ == '__main__':
    main()
