import numpy as np
import pandas as pd
import logging

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from visual_embeddings.arguments import get_args
from visual_embeddings.preprocessing import generate_all_offline_dataloaders, generate_online_dataloader
from visual_embeddings.models.embedding_network import *
from visual_embeddings.models.classification_network import *
from visual_embeddings.utils import *


def main():
    args = get_args() # Get arguments

    torch.set_num_threads(1) # Prevent error on KeyboardInterrupt with multiple GPUs
    if args.device == 'cuda': # Set correct default GPU
        torch.cuda.set_device(args.device_ids[0])

    # Create all required directories if not present
    make_dirs(args.project_dir, [args.checkpoints_dir])
    make_dirs(args.logs_dir)
    make_dirs(args.plots_dir)

    setup_logging(args.logs_dir) # Setup configuration for logging

    # Print all arguments
    global_vars = vars(args).copy()
    print_config(global_vars)

    if args.offline:
        train_loader, val_loader, test_loader = generate_all_offline_dataloaders(args)
    else:
        train_loader, transforms = generate_online_dataloader(args, args.num_train, 'train', None)
        val_loader = generate_online_dataloader(args, args.num_val, 'val', transforms)
        test_loader = generate_online_dataloader(args, args.num_test, 'test', transforms)

    # Declare Network and Hyperparameters
    logging.info('Creating models...')
    img_dim = train_loader.dataset.__getitem__(0)[0].shape[-1]
    num_outputs = len(args.time_buckets)
    if args.emb_model == 'cnn':
        in_dim, in_channels, out_dim = img_dim, args.num_frames*args.num_channels, 256
        embedding_hidden_size, classification_hidden_size = 256, 256
        embedding_network = CNNNetwork(in_dim, in_channels, embedding_hidden_size, out_dim, \
                                       use_pool=args.use_pool, use_res=args.use_res)
    elif args.emb_model == 'emb-cnn1':
        in_dim, out_dim = img_dim, 256
        embedding_size, embedding_hidden_size, classification_hidden_size = 8, 256, 256
        embedding_network = EmbeddingCNNNetwork1(in_dim, embedding_size, embedding_hidden_size, out_dim)
    elif args.emb_model == 'emb-cnn2':
        in_dim, out_dim = img_dim, 256
        embedding_size, embedding_hidden_size, classification_hidden_size = 8, 256, 256
        embedding_network = EmbeddingCNNNetwork2(in_dim, embedding_size, embedding_hidden_size, out_dim)
    elif args.emb_model == 'rel':
        out_dim = 512
        embedding_size, embedding_hidden_size, classification_hidden_size = 8, 512, 512
        embedding_network = RelativeNetwork(embedding_size, embedding_hidden_size, out_dim)
    else:
        raise ValueError('Unknown embedding network name "{}" passed.'.format(args.emb_model))
    classification_network = ClassificationNetwork(out_dim, classification_hidden_size, num_outputs)
    logging.info('Done.')

    logging.info(embedding_network)
    logging.info(classification_network)

    # Define criteria and optimizer
    criterion_train = nn.CrossEntropyLoss()
    criterion_test = nn.CrossEntropyLoss(reduction='sum')
    optimizer = optim.Adam(list(embedding_network.parameters()) + list(classification_network.parameters()), lr=args.lr)

    start_epoch = 0 # Initialize starting epoch number (used later if checkpoint loaded)
    stop_epoch = args.epochs+start_epoch # Store epoch upto which model is trained (used in case of KeyboardInterrupt)

    train_loss_history, train_accuracy_history = [], []
    val_loss_history, val_accuracy_history = [], []

    # Load model state dicts / models if required
    epoch_trained = 0
    if args.load_ckpt: # First check for state dicts
        embedding_network, classification_network, optimizer, train_loss_history, val_loss_history, \
        train_accuracy_history, val_accuracy_history, epoch_trained = \
            load_checkpoint(embedding_network, classification_network, optimizer, args)
    elif args.load_emb_ckpt and args.load_cls_ckpt: # Otherwise check for entire model
        embedding_network, epoch_trained = load_model(args, args.load_emb_ckpt)
        classification_network, epoch_trained_cls = load_model(args, args.load_cls_ckpt)
        assert epoch_trained == epoch_trained_cls, \
            'Mismatch in epochs trained for embedding network (={}) and classification network (={}).'\
            .format(epoch_trained, epoch_trained_cls)
    start_epoch = epoch_trained # Start from (epoch_trained+1) if checkpoint loaded

    # Parallelize models
    embedding_network = embedding_network.to(args.device)
    classification_network = classification_network.to(args.device)
    if args.device == 'cuda':
        logging.info('Using {} GPUs...'.format(len(args.device_ids)))
        embedding_network = nn.DataParallel(embedding_network, device_ids=args.device_ids)
        classification_network = nn.DataParallel(classification_network, device_ids=args.device_ids)
        logging.info('Done.')

    early_stopping = EarlyStopping(mode='maximize', min_delta=0.5, patience=10)
    best_epoch = start_epoch+1

    for epoch in range(start_epoch+1, args.epochs+start_epoch+1):
        try:
            train_losses = train(
                embedding_network=embedding_network,
                classification_network=classification_network,
                criterion=criterion_train,
                dataloader=train_loader,
                optimizer=optimizer,
                device=args.device,
                epoch=epoch
            )

            accuracy_val, val_loss = test(
                embedding_network=embedding_network,
                classification_network=classification_network,
                dataloader=val_loader,
                criterion=criterion_test,
                device=args.device
            )

            train_loss_history.extend(train_losses)
            val_loss_history.append(val_loss)

            accuracy_train, _ = test(embedding_network, classification_network, train_loader, criterion_test, args.device)
            train_accuracy_history.append(accuracy_train)
            val_accuracy_history.append(accuracy_val)

            logging.info('TRAIN Epoch: {}\tAverage loss: {:.4f}, Accuracy: {:.0f}%'\
                         .format(epoch, np.sum(train_losses), accuracy_train))
            logging.info('VAL   Epoch: {}\tAverage loss: {:.4f}, Accuracy: {:.0f}%\n'\
                         .format(epoch, val_loss, accuracy_val))

            if early_stopping.is_better(accuracy_val):
                logging.info('Saving current best model checkpoint...')
                save_checkpoint(embedding_network, classification_network, optimizer, train_loss_history, \
                                val_loss_history, train_accuracy_history, val_accuracy_history, args, epoch)
                logging.info('Done.')
                logging.info('Removing previous best model checkpoint...')
                remove_checkpoint(args, best_epoch)
                logging.info('Done.')
                best_epoch = epoch

            if early_stopping.stop(accuracy_val) or round(accuracy_val) == 100:
                logging.info('Stopping early after {} epochs.'.format(epoch))
                break

            stop_epoch = epoch
        except KeyboardInterrupt:
            logging.info('Keyboard Interrupted!')
            stop_epoch = epoch-1
            break

    # Save the model checkpoints
    logging.info('Dumping model and results...')
    print_config(global_vars) # Print all global variables before saving checkpointing
    save_checkpoint(embedding_network, classification_network, optimizer, train_loss_history, val_loss_history, \
                    train_accuracy_history, val_accuracy_history, args, stop_epoch)
    save_model(embedding_network, 'embedding_network', args, stop_epoch)
    save_model(classification_network, 'classification_network', args, stop_epoch)
    logging.info('Done.')

    title = '{}_{}'.format(args.emb_model, args.dataset_name)
    if len(train_loss_history) and len(val_loss_history):
        logging.info('Plotting and saving loss histories...')
        fig = plt.figure(figsize=(10,8))
        plt.plot(train_loss_history, alpha=0.5, color='blue', label='train')
        xticks = [epoch*len(train_loader) for epoch in range(1, len(val_loss_history)+1)]
        plt.plot(xticks, val_loss_history, alpha=0.5, color='orange', marker='x', label='test')
        plt.legend()
        plt.title('Loss vs. Iterations ({})'.format(args.dataset_name), fontsize=16, ha='center')
        plt.xlabel('Iterations', fontsize=16, ha='center')
        plt.ylabel('Loss', fontsize=16, ha='center')
        save_plot(args, fig, 'loss_vs_iterations_{}.eps'.format(title))
        logging.info('Done.')

    if len(train_accuracy_history) and len(val_accuracy_history):
        logging.info('Plotting and saving accuracy histories...')
        fig = plt.figure(figsize=(10,8))
        xticks = range(1, len(val_accuracy_history)+1)
        plt.plot(xticks, train_accuracy_history, alpha=0.5, color='blue', label='train')
        plt.plot(xticks, val_accuracy_history, alpha=0.5, color='orange', label='test')
        plt.legend()
        plt.title('Accuracy vs. Epochs ({})'.format(args.dataset_name), fontsize=16, ha='center')
        plt.xlabel('Epochs', fontsize=16, ha='center')
        plt.ylabel('Accuracy', fontsize=16, ha='center')
        save_plot(args, fig, 'accuracies_vs_epochs_{}.eps'.format(title))
        logging.info('Done.')

if __name__ == '__main__':
    main()
