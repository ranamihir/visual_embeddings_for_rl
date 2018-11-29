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
CHECKPOINTS_DIR, EMB_MODEL_CKPT = args.checkpoints_dir, args.load_emb_ckpt
assert EMB_MODEL_CKPT is not None, 'Model checkpoint not provided.'
DATASET_TYPE, DATASET, DATA_EXT = args.dataset_type, args.dataset, args.data_ext
EMB_MODEL = args.emb_model

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

NUM_CHANNELS = args.num_channels        # number of channels in each input image frame
USE_POOL = args.use_pool                # use pooling instead of strided convolutions
USE_RES = args.use_res                  # use residual layers

def main():
    torch.set_num_threads(1) # Prevent error on KeyboardInterrupt with multiple GPUs

    # Create all required directories if not present
    make_dirs(PROJECT_DIR, [LOGGING_DIR])
    make_dirs(os.path.join(PROJECT_DIR, DATA_DIR), [EMBEDDINGS_DIR])

    setup_logging(PROJECT_DIR, LOGGING_DIR) # Setup configuration for logging

    global_vars = globals().copy()
    print_config(global_vars) # Print all global variables defined above

    dataloader = generate_embedding_dataloader(PROJECT_DIR, DATA_DIR, DATASET_TYPE, DATASET, \
                                               EMB_MODEL, BATCH_SIZE, NUM_CHANNELS, DATA_EXT)

    # Load trained model
    logging.info('Loading trained model...')
    embedding_network = load_model(PROJECT_DIR, CHECKPOINTS_DIR, EMB_MODEL_CKPT)
    logging.info('Done.')
    logging.info(embedding_network)

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
                    DATASET, EMB_MODEL, USE_POOL, USE_RES)
    logging.info('Done.')

if __name__ == '__main__':
    main()
