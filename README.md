# Learning Visual Embeddings for Reinforcement Learning
Members:
  - Mihir Rana
  - Kenil Tanna


## Requirements
For ease of setup, we have created a [requirements.yaml](https://github.com/ranamihir/visual_embeddings_for_rl/blob/master/requirements.yaml) file which will create a conda environment with the name `visual_embeddings` and install most dependencies and requirements into that environment.

To do this, install [Anaconda](https://www.anaconda.com/) and run the following commands:
  - ```bash
    conda env create -f requirements.yaml
    source activate visual_embeddings
    ```
Additionally:
  - Install [OpenAI baselines](https://github.com/openai/baselines)
  - Install [MiniGrid](https://github.com/maximecb/gym-minigrid):
    <br>
    ```bash
    git clone https://github.com/maximecb/gym-minigrid.git
    cd gym-minigrid
    pip install -e .
    cd ..
    ```

## Installation
Again, for simplicity, we have created a module with the name `visual_embeddings` which can be installed directly into PyPI by running the following command from the main project directory:
```
pip install -e .
```

## Usage
```
usage: main.py [-h] [--project-dir PROJECT_DIR] [--data-dir DATA_DIR]
               [--plots-dir PLOTS_DIR] [--logs-dir LOGS_DIR]
               [--checkpoints-dir CHECKPOINTS_DIR]
               [--embeddings-dir EMBEDDINGS_DIR] [--dataset-type DATASET_TYPE]
               [--dataset DATASET] [--data-ext DATA_EXT] [--offline] [--force]
               [--cpu] [--cuda] [--device DEVICE] [--device-ids DEVICE_IDS]
               [--parallel] [--emb-model EMB_MODEL]
               [--load-ckpt LOAD_CHECKPOINT] [--load-emb-ckpt LOAD_EMB_CKPT]
               [--load-cls-ckpt LOAD_CLS_CKPT] [--batch-size BATCH_SIZE]
               [--epochs EPOCHS] [--lr LR] [--flatten] [--num-train NUM_TRAIN]
               [--num-frames NUM_FRAMES_IN_STACK]
               [--num-channels NUM_CHANNELS]
               [--num-pairs NUM_PAIRS_PER_EXAMPLE] [--use-pool] [--use-res]

Learning Visual Embeddings for Reinforcement Learning

optional arguments:
  -h, --help                          show this help message and exit
  --project-dir PROJECT_DIR           path to project directory
  --data-dir DATA_DIR                 path to data directory, default="data/"
  --plots-dir PLOTS_DIR               path to plots directory, default="logs"plots/
  --logs-dir LOGS_DIR                 path to logs directory, default="logs/"
  --checkpoints-dir CHECKPOINTS_DIR   path to checkpoints directory, default="checkpoints/"
  --embeddings-dir EMBEDDINGS_DIR     path to embeddings directory, default="checkpoints/embeddings/"
  --dataset-type DATASET_TYPE         name of PyTorch Dataset to use
                                      maze | fixed_mmnist | random_mmnist, default=maze
  --dataset DATASET                   name of dataset file in "data" directory
                                      mnist_test_seq | moving_bars_20_121 | etc., default=all_mazes_16_3_6
  --data-ext DATA_EXT                 extension of dataset file in data directory
  --offline                           use offline preprocessing of data loader
  --force                             overwrites all existing dumped data sets (if used with `--offline`)
  --cpu                               use CPU
  --cuda                              use CUDA, default id: 0
  --device                            cuda | cpu, default=cuda
                                      device to train on
  --device-ids DEVICE_IDS             IDs of GPUs to use
  --parallel                          use all GPUs available
  --emb-model EMB_MODEL               type of embedding network
  --load-ckpt LOAD_CHECKPOINT         checkpoint file to load
  --load-emb-ckpt LOAD_EMB_CKPT       embedding network checkpoint file to load
  --load-cls-ckpt LOAD_CLS_CKPT       classification network checkpoint file to load
  --batch-size BATCH_SIZE             input batch size, default=64
  --epochs EPOCHS                     number of epochs, default=10
  --lr LR                             learning rate, default=1e-4
  --flatten                           flatten data into 1 long video
  --num-train NUM_TRAIN               number of paired training examples, default=500000
  --num-frames NUM_FRAMES_IN_STACK    number of stacked frames, default=2
  --num-channels NUM_CHANNELS         number of channels in input image, default=1
  --num-pairs NUM_PAIRS_PER_EXAMPLE   number of pairs per video, default=5
  --use-pool                          use max pooling instead of strided convolutions in embedding network
  --use-res                           use residual layers in embedding network
```

## Training

### Embedding Generation

#### Minigrid Maze
```
python main.py --dataset all_mazes_10000_16_3_6 --dataset-type maze --epochs 15 --emb-model emb-cnn1 --num-frames 1  --num-channels 3 --flatten
```
<img src="https://github.com/ranamihir/visual_embeddings_for_rl/blob/master/material/mazes.gif" width="512" height="512" />

#### Moving MNIST (Random Trajectories)
```
python main.py --dataset moving_mnist --dataset-type random_mmnist --data-ext .h5 --num-frames 4 --use-pool
```

#### Moving MNIST (Fixed Trajectories)
```
python main.py --dataset mnist_test_seq --dataset-type fixed_mmnist --data-ext .npy --num-frames 2 --use-pool
```

#### Moving Bars
```
python generate_lines_data.py --seq-len 50 --img-dim 121
python main.py --dataset moving_bars_50_121 --dataset-type fixed_mmnist --data-ext .npy --num-frames 4
```
