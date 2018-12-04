import numpy as np
import pandas as pd
import logging

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from capstone_project.arguments import get_args
from capstone_project.preprocessing import generate_all_offline_dataloaders, generate_online_dataloader
from capstone_project.models.embedding_network import *
from capstone_project.models.classification_network import *
from capstone_project.utils import *


args = get_args()


# Globals
PROJECT_DIR = args.project_dir
DATA_DIR,  PLOTS_DIR, LOGGING_DIR = args.data_dir, 'plots', 'logs'
CHECKPOINTS_DIR, CHECKPOINT_FILE = args.checkpoints_dir, args.load_ckpt
EMB_MODEL_CKPT, CLS_MODEL_CKPT = args.load_emb_ckpt, args.load_cls_ckpt
DATASET_TYPE, DATASET, DATA_EXT = args.dataset_type, args.dataset, args.data_ext
OFFLINE = args.offline
EMB_MODEL = args.emb_model
FORCE = args.force

TEST_SIZE, VAL_SIZE = 0.2, 0.2
if not OFFLINE:
    NUM_TRAIN = args.num_train
    TRAIN_SIZE = 1 - TEST_SIZE - VAL_SIZE
    NUM_TEST, NUM_VAL = int((TEST_SIZE/TRAIN_SIZE)*NUM_TRAIN), int((VAL_SIZE/TRAIN_SIZE)*NUM_TRAIN)

BATCH_SIZE = args.batch_size    # input batch size for training
N_EPOCHS = args.epochs          # number of epochs to train
LR = args.lr                    # learning rate
NGPU = args.ngpu                # number of GPUs
PARALLEL = args.parallel        # use all GPUs

TOTAL_GPUs = torch.cuda.device_count() # Number of total GPUs available

if NGPU:
    assert TOTAL_GPUs >= NGPU, '{} GPUs not available! Only {} GPU(s) available'.format(NGPU, TOTAL_GPUs)

DEVICE = args.device if args.device else 'cuda' if torch.cuda.is_available() else 'cpu'
if args.device_id and 'cuda' in DEVICE:
    DEVICE_ID = args.device_id
    torch.cuda.set_device(DEVICE_ID)

FLATTEN = args.flatten                      # flatten data into 1 long video
NUM_FRAMES_IN_STACK = args.num_frames       # number of (total) frames to concatenate for each video
NUM_CHANNELS = args.num_channels            # number of channels in each input image frame
NUM_PAIRS_PER_EXAMPLE = args.num_pairs      # number of pairs to generate for given video and time difference
USE_POOL = args.use_pool                    # use pooling instead of strided convolutions
USE_RES = args.use_res                      # use residual layers
TIME_BUCKETS = [[0], [1], [2], [3,4], range(5,11,1)]

def main():
    torch.set_num_threads(1) # Prevent error on KeyboardInterrupt with multiple GPUs

    make_dirs(PROJECT_DIR, [CHECKPOINTS_DIR, PLOTS_DIR, LOGGING_DIR]) # Create all required directories if not present
    setup_logging(PROJECT_DIR, LOGGING_DIR) # Setup configuration for logging

    global_vars = globals().copy()
    print_config(global_vars) # Print all global variables defined above

    if OFFLINE:
        train_loader, val_loader, test_loader = generate_all_offline_dataloaders(PROJECT_DIR, DATA_DIR, PLOTS_DIR, DATASET, \
                                                                                TIME_BUCKETS, BATCH_SIZE, NUM_PAIRS_PER_EXAMPLE, \
                                                                                NUM_FRAMES_IN_STACK, DATA_EXT, FORCE)
    else:
        train_loader, transforms = generate_online_dataloader(PROJECT_DIR, DATA_DIR, PLOTS_DIR, DATASET_TYPE, DATASET, \
                                                            NUM_TRAIN, 'train', TIME_BUCKETS, EMB_MODEL, BATCH_SIZE, \
                                                            NUM_FRAMES_IN_STACK, NUM_CHANNELS, DATA_EXT, FLATTEN, None, FORCE)
        val_loader = generate_online_dataloader(PROJECT_DIR, DATA_DIR, PLOTS_DIR, DATASET_TYPE, DATASET, NUM_VAL, 'val', \
                                        TIME_BUCKETS, EMB_MODEL, BATCH_SIZE, NUM_FRAMES_IN_STACK, NUM_CHANNELS, \
                                        DATA_EXT, FLATTEN, transforms, FORCE)
        test_loader = generate_online_dataloader(PROJECT_DIR, DATA_DIR, PLOTS_DIR, DATASET_TYPE, DATASET, NUM_TEST, 'test', \
                                        TIME_BUCKETS, EMB_MODEL, BATCH_SIZE, NUM_FRAMES_IN_STACK, NUM_CHANNELS, \
                                        DATA_EXT, FLATTEN, transforms, FORCE)

    # Declare Network and Hyperparameters
    logging.info('Creating models...')
    img_dim = train_loader.dataset.__getitem__(0)[0].shape[-1]
    num_outputs = len(TIME_BUCKETS)
    if EMB_MODEL == 'cnn':
        in_dim, in_channels, out_dim = img_dim, NUM_FRAMES_IN_STACK*NUM_CHANNELS, 256
        embedding_hidden_size, classification_hidden_size = 256, 256
        embedding_network = CNNNetwork(in_dim, in_channels, embedding_hidden_size, out_dim, use_pool=USE_POOL, use_res=USE_RES)
    elif EMB_MODEL == 'emb-cnn1':
        in_dim, out_dim = img_dim, 256
        embedding_size, embedding_hidden_size, classification_hidden_size = 8, 256, 256
        embedding_network = EmbeddingCNNNetwork1(in_dim, embedding_size, embedding_hidden_size, out_dim)
    elif EMB_MODEL == 'emb-cnn2':
        in_dim, out_dim = img_dim, 256
        embedding_size, embedding_hidden_size, classification_hidden_size = 8, 256, 256
        embedding_network = EmbeddingCNNNetwork2(in_dim, embedding_size, embedding_hidden_size, out_dim)
    elif EMB_MODEL == 'rel':
        out_dim = 512
        embedding_size, embedding_hidden_size, classification_hidden_size = 8, 512, 512
        embedding_network = RelativeNetwork(embedding_size, embedding_hidden_size, out_dim)
    else:
        raise ValueError('Unknown embedding network name "{}" passed.'.format(EMB_MODEL))
    classification_network = ClassificationNetwork(out_dim, classification_hidden_size, num_outputs)
    logging.info('Done.')
    logging.info(embedding_network)
    logging.info(classification_network)

    # Define criteria and optimizer
    criterion_train = nn.CrossEntropyLoss()
    criterion_test = nn.CrossEntropyLoss(reduction='sum')
    optimizer = optim.Adam(list(embedding_network.parameters()) + list(classification_network.parameters()), lr=LR)

    start_epoch = 0 # Initialize starting epoch number (used later if checkpoint loaded)
    stop_epoch = N_EPOCHS+start_epoch # Store epoch upto which model is trained (used in case of KeyboardInterrupt)

    train_loss_history, train_accuracy_history = [], []
    val_loss_history, val_accuracy_history = [], []

    # Load model state dicts / models if required
    if CHECKPOINT_FILE: # First check for state dicts
        embedding_network, classification_network, optimizer, train_loss_history, val_loss_history, \
        train_accuracy_history, val_accuracy_history, epoch_trained = \
            load_checkpoint(embedding_network, classification_network, optimizer, CHECKPOINT_FILE, \
                            PROJECT_DIR, CHECKPOINTS_DIR, DEVICE)
    elif EMB_MODEL_CKPT or CLS_MODEL_CKPT: # Otherwise check for entire model
        if EMB_MODEL_CKPT:
            embedding_network, epoch_trained_emb = load_model(PROJECT_DIR, CHECKPOINTS_DIR, EMB_MODEL_CKPT)
        if CLS_MODEL_CKPT:
            classification_network, epoch_trained_cls = load_model(PROJECT_DIR, CHECKPOINTS_DIR, CLS_MODEL_CKPT)
            assert epoch_trained_emb == epoch_trained_cls, \
                'Mismatch in epochs trained for embedding network (={}) and classification network (={}).'\
                .format(epoch_trained_emb, epoch_trained_cls)
    start_epoch = epoch_trained # Start from (epoch_trained+1) if checkpoint loaded

    # Check if model is to be parallelized
    if TOTAL_GPUs > 1 and (PARALLEL or NGPU):
        DEVICE_IDs = range(TOTAL_GPUs) if PARALLEL else range(NGPU)
        logging.info('Using {} GPUs...'.format(len(DEVICE_IDs)))
        embedding_network = nn.DataParallel(embedding_network, device_ids=DEVICE_IDs)
        classification_network = nn.DataParallel(classification_network, device_ids=DEVICE_IDs)
        logging.info('Done.')
    embedding_network = embedding_network.to(DEVICE)
    classification_network = classification_network.to(DEVICE)

    early_stopping = EarlyStopping(mode='maximize', min_delta=0.5, patience=10)
    best_epoch = start_epoch+1

    for epoch in range(start_epoch+1, N_EPOCHS+start_epoch+1):
        try:
            train_losses = train(
                embedding_network=embedding_network,
                classification_network=classification_network,
                criterion=criterion_train,
                dataloader=train_loader,
                optimizer=optimizer,
                device=DEVICE,
                epoch=epoch
            )

            accuracy_val, val_loss = test(
                embedding_network=embedding_network,
                classification_network=classification_network,
                dataloader=val_loader,
                criterion=criterion_test,
                device=DEVICE
            )

            train_loss_history.extend(train_losses)
            val_loss_history.append(val_loss)

            accuracy_train, _ = test(embedding_network, classification_network, train_loader, criterion_test, DEVICE)
            train_accuracy_history.append(accuracy_train)
            val_accuracy_history.append(accuracy_val)

            logging.info('TRAIN Epoch: {}\tAverage loss: {:.4f}, Accuracy: {:.0f}%'\
                         .format(epoch, np.sum(train_losses), accuracy_train))
            logging.info('VAL   Epoch: {}\tAverage loss: {:.4f}, Accuracy: {:.0f}%\n'\
                         .format(epoch, val_loss, accuracy_val))

            if early_stopping.is_better(accuracy_val):
                logging.info('Saving current best model checkpoint...')
                save_checkpoint(embedding_network, classification_network, optimizer, train_loss_history, \
                                val_loss_history, train_accuracy_history, val_accuracy_history, epoch, DATASET, \
                                EMB_MODEL, NUM_FRAMES_IN_STACK, NUM_PAIRS_PER_EXAMPLE, PROJECT_DIR, \
                                CHECKPOINTS_DIR, USE_POOL, USE_RES, PARALLEL or NGPU)
                logging.info('Done.')
                logging.info('Removing previous best model checkpoint...')
                remove_checkpoint(DATASET, EMB_MODEL, NUM_FRAMES_IN_STACK, NUM_PAIRS_PER_EXAMPLE, \
                                  PROJECT_DIR, CHECKPOINTS_DIR, best_epoch, USE_POOL, USE_RES)
                logging.info('Done.')
                best_epoch = epoch

            if early_stopping.stop(accuracy_val) or round(accuracy_val) == 100:
                logging.info('Stopping early after {} epochs.'.format(epoch))
                stop_epoch = epoch
                break
        except KeyboardInterrupt:
            logging.info('Keyboard Interrupted!')
            stop_epoch = epoch-1
            break

    # Save the model checkpoints
    logging.info('Dumping model and results...')
    print_config(global_vars) # Print all global variables before saving checkpointing
    save_checkpoint(embedding_network, classification_network, optimizer, train_loss_history, val_loss_history, \
                    train_accuracy_history, val_accuracy_history, stop_epoch, DATASET, EMB_MODEL, NUM_FRAMES_IN_STACK, \
                    NUM_PAIRS_PER_EXAMPLE, PROJECT_DIR, CHECKPOINTS_DIR, USE_POOL, USE_RES, PARALLEL or NGPU)
    save_model(embedding_network, 'embedding_network', stop_epoch, DATASET, EMB_MODEL, \
               NUM_FRAMES_IN_STACK, NUM_PAIRS_PER_EXAMPLE, PROJECT_DIR, CHECKPOINTS_DIR, USE_POOL, USE_RES)
    save_model(classification_network, 'classification_network', stop_epoch, DATASET, EMB_MODEL, \
               NUM_FRAMES_IN_STACK, NUM_PAIRS_PER_EXAMPLE, PROJECT_DIR, CHECKPOINTS_DIR, USE_POOL, USE_RES)
    logging.info('Done.')

    if len(train_loss_history) and len(val_loss_history):
        logging.info('Plotting and saving loss histories...')
        fig = plt.figure(figsize=(10,8))
        plt.plot(train_loss_history, alpha=0.5, color='blue', label='train')
        xticks = [epoch*len(train_loader) for epoch in range(1, len(val_loss_history)+1)]
        plt.plot(xticks, val_loss_history, alpha=0.5, color='orange', label='test')
        plt.legend()
        plt.title('Loss vs. Iterations')
        save_plot(PROJECT_DIR, PLOTS_DIR, fig, 'loss_vs_iterations.png')
        logging.info('Done.')

    if len(train_accuracy_history) and len(val_accuracy_history):
        logging.info('Plotting and saving accuracy histories...')
        fig = plt.figure(figsize=(10,8))
        plt.plot(train_accuracy_history, alpha=0.5, color='blue', label='train')
        plt.plot(val_accuracy_history, alpha=0.5, color='orange', label='test')
        plt.legend()
        plt.title('Accuracy vs. Iterations')
        save_plot(PROJECT_DIR, PLOTS_DIR, fig, 'accuracies_vs_iterations.png')
        logging.info('Done.')

if __name__ == '__main__':
    main()
