# DS-GA 1006 Capstone Project and Presentation

# Learning Visual Embeddings
Members:
  - Mihir Rana
  - Kenil Tanna


## Requirements
For ease of setup, we have created a [requirements.yaml](https://github.com/ranamihir/capstone_project/blob/master/requirements.yaml) file which will create a conda environment with the name `capstone_project` and install all dependencies and requirements into that environment. To do this:
  - Install Anaconda and run:
```
conda env create -f requirements.yaml
```
  - Optionally, if you want to run it on a GPU, install CUDA and cuDNN

## Installation
Again, for simplicity, we have created a module with the name `capstone_project` which can be installed directly into pip by running the following command from the main project directory:
```
pip install -e .
```

## Usage
```
usage: main.py [-h] [--project-dir PROJECT_DIR] [--dataset DATASET]
               [--data-dir DATA_DIR] [--batch-size BATCH_SIZE]
               [--epochs EPOCHS] [--device DEVICE] [--device-id DEVICE_ID]
               [--ngpu NGPU] [--lr LR] [--force]

optional arguments:
  -h, --help                      show this help message and exit
  --project-dir PROJECT_DIR       path to project directory
  --dataset DATASET               name of dataset file in 'data' directory
                                  mnist_test_seq.py | cifar10.py, default=mnist_test_seq.py
  --data-dir DATA_DIR             path to data directory (used if different from 'data')
  --batch-size BATCH_SIZE         input batch size, default=64
  --epochs EPOCHS                 number of epochs, default=10
  --lr LR                         learning rate, default=1e-4
  --device                        cuda | cpu, default=cuda
                                  device to train on
  --device-id DEVICE_ID           device id of gpu
  --ngpu NGPU                     number of GPUs to use
  --force                         overwrites all existing data
