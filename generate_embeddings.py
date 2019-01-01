import numpy as np
import pandas as pd
import os
import logging

import torch
import torch.nn as nn

from visual_embeddings.arguments import get_args
from visual_embeddings.preprocessing import generate_embedding_dataloader
from visual_embeddings.models.embedding_network import *
from visual_embeddings.utils import *


def main():
    args = get_args() # Get arguments

    torch.set_num_threads(1) # Prevent error on KeyboardInterrupt with multiple GPUs
    if args.device == 'cuda': # Set correct default GPU
        torch.cuda.set_device(args.device_ids[0])

    assert args.load_emb_ckpt is not None, 'Model checkpoint not provided.'
    args.batch_size = 1     # Each video separately

    # Create all required directories if not present
    make_dirs(args.logs_dir)
    make_dirs(args.embeddings_dir)

    setup_logging(args.logs_dir) # Setup configuration for logging

    # Print all arguments
    global_vars = vars(args).copy()
    print_config(global_vars)

    dataloader = generate_embedding_dataloader(args)

    # Load trained model
    logging.info('Loading trained model...')
    embedding_network, _ = load_model(args, args.load_emb_ckpt)
    logging.info('Done.')
    logging.info(embedding_network)

    # Parallelize models
    embedding_network = embedding_network.to(args.device)
    if args.device == 'cuda':
        logging.info('Using {} GPU(s)...'.format(len(args.device_ids)))
        embedding_network = nn.DataParallel(embedding_network, device_ids=args.device_ids)
        logging.info('Done.')

    try:
        embeddings = get_embeddings(
            embedding_network=embedding_network,
            dataloader=dataloader,
            device=args.device
        )
    except KeyboardInterrupt:
        logging.info('Keyboard Interrupted!')

    # Save the model checkpoints
    logging.info('Dumping embeddings...')
    save_embeddings(args, embeddings)
    logging.info('Done.')

if __name__ == '__main__':
    main()
