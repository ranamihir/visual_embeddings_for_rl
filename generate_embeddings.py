import numpy as np
import pandas as pd
import os
import logging

import torch
import torch.nn as nn

from capstone_project.arguments import get_args
from capstone_project.preprocessing import generate_embedding_dataloader
from capstone_project.models.embedding_network import *
from capstone_project.utils import *


args = get_args()


# Globals
PROJECT_DIR = args.project_dir
DATA_DIR, EMBEDDINGS_DIR, LOGGING_DIR = args.data_dir, 'embeddings', 'logs'
CHECKPOINTS_DIR, CHECKPOINT_FILE = args.checkpoints_dir, args.load_ckpt
assert CHECKPOINT_FILE is not None, 'Model checkpoint not provided.'
DATASET_TYPE, DATASET, DATA_EXT = args.dataset_type, args.dataset, args.data_ext
MODEL = args.model

BATCH_SIZE = 1                  # each video separately
NGPU = args.ngpu                # number of GPUs
PARALLEL = args.parallel        # use all GPUs

TOTAL_GPUs = torch.cuda.device_count() # Number of total GPUs available

if NGPU:
    assert TOTAL_GPUs >= NGPU, '{} GPUs not available! Only {} GPU(s) available'.format(NGPU, TOTAL_GPUs)

DEVICE = args.device if args.device else 'cuda' if torch.cuda.is_available() else 'cpu'
if args.device_id and 'cuda' in DEVICE:
    DEVICE_ID = args.device_id
    torch.cuda.set_device(DEVICE_ID)

NUM_CHANNELS = args.num_channels            # number of channels in each input image frame
USE_POOL = args.use_pool                    # use pooling instead of strided convolutions
USE_RES = args.use_res                      # use residual layers

def main():
    torch.set_num_threads(1) # Prevent error on KeyboardInterrupt with multiple GPUs

    # Create all required directories if not present
    make_dirs(PROJECT_DIR, [LOGGING_DIR])
    make_dirs(os.path.join(PROJECT_DIR, DATA_DIR), [EMBEDDINGS_DIR])

    setup_logging(PROJECT_DIR, LOGGING_DIR) # Setup configuration for logging

    global_vars = globals().copy()
    print_config(global_vars) # Print all global variables defined above

    dataloader = generate_embedding_dataloader(PROJECT_DIR, DATA_DIR, DATASET_TYPE, DATASET, \
                                               MODEL, BATCH_SIZE, NUM_CHANNELS, DATA_EXT)

    # Declare Network and Hyperparameters
    logging.info('Creating models...')
    img_dim = dataloader.dataset.__getitem__(0)[0].shape[-1]
    if MODEL == 'cnn':
        in_dim, in_channels, out_dim = img_dim, NUM_FRAMES_IN_STACK*NUM_CHANNELS, 256
        embedding_hidden_size = 256
        embedding_network = CNNNetwork(in_dim, in_channels, embedding_hidden_size, out_dim, use_pool=USE_POOL, use_res=USE_RES)
    elif MODEL == 'emb-cnn1':
        in_dim, out_dim = img_dim, 256
        embedding_size, embedding_hidden_size = 8, 256
        embedding_network = EmbeddingCNNNetwork1(in_dim, embedding_size, embedding_hidden_size, out_dim)
    elif MODEL == 'emb-cnn2':
        in_dim, out_dim = img_dim, 256
        embedding_size, embedding_hidden_size = 8, 256
        embedding_network = EmbeddingCNNNetwork2(in_dim, embedding_size, embedding_hidden_size, out_dim)
    elif MODEL == 'rel':
        out_dim = 512
        embedding_size, embedding_hidden_size = 8, 512
        embedding_network = RelativeNetwork(embedding_size, embedding_hidden_size, out_dim)
    else:
        raise ValueError('Unknown model name "{}" passed.'.format(MODEL))
    logging.info('Done.')
    logging.info(embedding_network)

    # Load trained model state dicts
    embedding_network = load_checkpoint(embedding_network, None, None, CHECKPOINT_FILE, \
                                        PROJECT_DIR, CHECKPOINTS_DIR, DEVICE)[0]

    # Check if model is to be parallelized
    if TOTAL_GPUs > 1 and (PARALLEL or NGPU):
        DEVICE_IDs = range(TOTAL_GPUs) if PARALLEL else range(NGPU)
        logging.info('Using {} GPUs...'.format(len(DEVICE_IDs)))
        embedding_network = nn.DataParallel(embedding_network, device_ids=DEVICE_IDs)
        logging.info('Done.')
    embedding_network = embedding_network.to(DEVICE)

    try:
        embeddings = get_embeddings(
            embedding_network=embedding_network,
            dataloader=dataloader,
            device=DEVICE
        )
    except KeyboardInterrupt:
        logging.info('Keyboard Interrupted!')

    # Save the model checkpoints
    logging.info('Dumping embeddings...')
    save_embeddings(embeddings, PROJECT_DIR, DATA_DIR, EMBEDDINGS_DIR, \
                    DATASET, MODEL, USE_POOL, USE_RES)
    logging.info('Done.')

if __name__ == '__main__':
    main()
