import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import make_grid
from torch.utils.data.sampler import Sampler

from capstone_project.preprocessing import generate_dataloader
from capstone_project.models.embedding_network import EmbeddingNetwork
from capstone_project.models.classification_network import ClassificationNetwork
from capstone_project.utils import train, test, accuracy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchviz import make_dot, make_dot_from_trace


# Options
DATASET = 'moving_mnist'
TEST_SIZE, VAL_SIZE = 0.2, 0.2
BATCH_SIZE = 64   # input batch size for training
N_EPOCHS = 10       # number of epochs to train
LR = 0.01        # learning rate
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    data = np.load('/home/mihir/Desktop/GitHub/nyu/capstone_project/data/mnist_test_seq.npy')
    data = np.swapaxes(data, 0, 1)
    train_loader, val_loader, test_loader = generate_dataloader(data, TEST_SIZE, VAL_SIZE, BATCH_SIZE)

    if DATASET == 'moving_mnist':
        num_inputs, n_channels = 64, 1
        num_outputs = 6
    elif DATASET == 'cifar10':
        num_inputs, n_channels = 32, 3
        num_outputs = 10

    train_loss_history = []
    test_loss_history = []
    embedding_network = EmbeddingNetwork(num_inputs, num_outputs).to(DEVICE)
    classification_network = ClassificationNetwork(num_inputs, num_outputs).to(DEVICE)

    criterion_train = nn.CrossEntropyLoss()
    criterion_test = nn.CrossEntropyLoss(reduction='sum')
    optimizer = optim.SGD(list(embedding_network.parameters()) + list(classification_network.parameters()), lr=LR)

    for epoch in range(1, N_EPOCHS+1):
        train_loss = train(
            embedding_network=embedding_network,
            classification_network=classification_network,
            criterion=criterion_train,
            dataloader=train_loader,
            optimizer=optimizer,
            device=DEVICE,
            epoch=epoch
        )

        test_loss, test_pred, test_true = test(
            network=network,
            criterion=criterion_test,
            dataloader=test_loader,
            device=DEVICE
        )

        accuracy_train = accuracy(embedding_network, classification_network, train_loader, criterion_test, DEVICE)
        accuracy_test = accuracy(embedding_network, classification_network, test_loader, criterion_test, DEVICE)
        train_loss_history.append(train_loss)
        test_loss_history.append(test_loss)

        print('TRAIN Epoch: {}\tAverage loss: {:.4f}, Accuracy: {:.0f}%'.format(epoch, train_loss, accuracy_train))
        print('TEST  Epoch: {}\tAverage loss: {:.4f}, Accuracy: {:.0f}%\n'.format(epoch, test_loss, accuracy_test))


    total_parameters_dict = dict(embedding_network.named_parameters())
    total_parameters_dict.update(dict(classification_network.named_parameters()))
    embedding_output1 = embedding_network(train_loader.dataset[0][0].to(DEVICE))
    embedding_output2 = embedding_network(train_loader.dataset[1][0].to(DEVICE))
    classification_input = torch.dot(embedding_output1, embedding_output2)
    classification_output = classification_network(classification_input)
    make_dot(classification_output, params=total_parameters_dict)

    with torch.onnx.set_training(network, False):
        embedding_output1 = embedding_network(train_loader.dataset[0][0].to(DEVICE))
        embedding_output2 = embedding_network(train_loader.dataset[1][0].to(DEVICE))
        classification_input = torch.dot(embedding_output1, embedding_output2)
        trace, _ = torch.jit.get_trace_graph(classification_network, args=(classification_input,))
    make_dot_from_trace(trace)

    loss_history_df = pd.DataFrame({
        'train': train_loss_history,
        'test': test_loss_history,
    })
    loss_history_df.plot(alpha=0.5, figsize=(10,8))

if __name__ == '__main__':
    main()
