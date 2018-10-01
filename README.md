# DS-GA 1006 Capstone Project and Presentation


## Requirements
For ease of setup, we have created a [requirement.yaml](https://github.com/ranamihir/capstone_project/blob/master/requirements.yaml) file which will create a conda environment with the name `capstone_project` and install all dependencies and requirements into that environment. To do this:
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
               [--batch-size BATCH_SIZE] [--epochs EPOCHS]
               [--lr LR] [--device DEVICE]

optional arguments:
  -h, --help                      show this help message and exit
  --project-dir PROJECT_DIR       path to project directory
  --dataset DATASET               mnist_test_seq.py | cifar10.py, default=mnist_test_seq.py
                                  name of dataset file in 'data' directory
  --batch-size BATCH_SIZE
                                  input batch size, default=16
  --epochs EPOCHS                 number of epochs to train for, default=50
  --lr LR                         learning rate, default=0.01
  --device                        cuda | cpu, default=cuda
                                  device to train on
