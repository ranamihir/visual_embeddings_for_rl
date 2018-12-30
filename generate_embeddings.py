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


args = get_args()

assert args.load_emb_ckpt is not None, 'Model checkpoint not provided.'
args.batch_size = 1     # Each video separately

def main():
    torch.set_num_threads(1) # Prevent error on KeyboardInterrupt with multiple GPUs

    # Create all required directories if not present
    make_dirs(args.logs_dir)
    make_dirs(args.embeddings_dir)

    # Setup configuration for logging
    setup_logging(args.logs_dir)

    global_vars = vars(args).copy()
    print_config(global_vars) # Print all arguments

    dataloader = generate_embedding_dataloader(args)

    # Load trained model
    logging.info('Loading trained model...')
    embedding_network, _ = load_model(args, args.load_emb_ckpt)
    logging.info('Done.')
    logging.info(embedding_network)

    # Check if model is to be parallelized
    if args.total_gpus > 1 and (args.parallel or args.ngpu):
        args.device_ids = range(args.total_gpus) if args.parallel else range(args.ngpu)
        logging.info('Using {} GPUs...'.format(len(args.device_ids)))
        embedding_network = nn.DataParallel(embedding_network, device_ids=args.device_ids)
        logging.info('Done.')
    embedding_network = embedding_network.to(args.device)

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
