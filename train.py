import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import make_grid
from torch.utils.data.sampler import Sampler

from preprocessing import generate_dataloader
from embedding_network import EmbeddingNetwork
from classification_network import ClassificationNetwork
from utils import train,test,accuracy,imshow

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchviz import make_dot, make_dot_from_trace


# options
DATASET = 'moving_mnist'
TEST_SIZE, VAL_SIZE = 0.2, 0.2
BATCH_SIZE = 64   # input batch size for training
N_EPOCHS = 10       # number of epochs to train
LR = 0.01        # learning rate
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


data = np.load('/home/mihir/Desktop/GitHub/nyu/capstone_project/data/mnist_test_seq.npy')
data = np.swapaxes(data, 0, 1)


train_loader, val_loader, test_loader = generate_dataloader(data, TEST_SIZE, VAL_SIZE, BATCH_SIZE)


# def imshow(data_loader):
#     data_iter = iter(data_loader)
#     images = data_iter.next()

#     images = make_grid(images[0].reshape(-1, 1, 64, 64), nrow=10)
#     np_image = images.numpy()

#     plt.figure(figsize=(50, 20))
#     plt.imshow(np.transpose(np_image, axes=(1, 2, 0)))


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


# def train(embedding_network, classification_network, dataloader, criterion, optimizer, epoch):
#     embedding_network.train()
#     classification_network.train()
#     loss_train = 0.
#     for batch_idx, (x1, x2, y) in enumerate(dataloader):
#         x1, x2, y = Variable(x1).to(DEVICE), Variable(x2).to(DEVICE), Variable(y).to(DEVICE)
#         optimizer.zero_grad()
#         embedding_output1 = embedding_network(x1)
#         embedding_output2 = embedding_network(x2)
#         classification_input = torch.dot(embedding_output1, embedding_output2)
#         classification_output = classification_network(classification_input)
#         loss = criterion(classification_output, y)
#         loss.backward()
#         optimizer.step()

#         # Accurately compute loss, because of different batch size
#         loss_train += loss.item() * len(x) / len(dataloader.dataset)

#         if batch_idx % 100 == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(x), len(dataloader.dataset),
#                 100. * batch_idx / len(dataloader), loss.item()))

#     optimizer.zero_grad()
#     return loss_train

# def test(embedding_network, classification_network, dataloader, criterion):
#     embedding_network.eval()
#     classification_network.eval()
#     loss_test = 0.
#     y_ls = []
#     output_ls = []
#     with torch.no_grad():
#         for batch_idx, (x1, x2, y) in enumerate(dataloader):
#             x1, x2, y = Variable(x1).to(DEVICE), Variable(x2).to(DEVICE), Variable(y).to(DEVICE)
#             embedding_output1 = embedding_network(x1)
#             embedding_output2 = embedding_network(x2)
#             classification_input = torch.dot(embedding_output1, embedding_output2)
#             classification_output = classification_network(classification_input)
#             loss = criterion(classification_output, y)

#             # Accurately compute loss, because of different batch size
#             loss_test += loss.item() / len(dataloader.dataset)

#             output_ls.append(classification_output)
#             y_ls.append(y)
#     optimizer.zero_grad()
#     return loss_test, torch.cat(output_ls, dim=0), torch.cat(y_ls, dim=0)

# def accuracy(embedding_network, classification_network, dataloader, criterion):
#     _, y_predicted, y_true = test(
#         embedding_network=embedding_network,
#         classification_network=classification_network,
#         dataloader=dataloader,
#         criterion=criterion
#     )
#     y_predicted = y_predicted.max(1)[1]
#     return 100*y_predicted.eq(y_true.data.view_as(y_predicted)).float().mean().item()


for epoch in range(1, N_EPOCHS+1):
    train_loss = train(
        embedding_network=embedding_network,
        classification_network=classification_network,
        criterion=criterion_train,
        dataloader=train_loader,
        optimizer=optimizer,
        epoch=epoch
    )

    test_loss, test_pred, test_true = test(
        network=network,
        criterion=criterion_test,
        dataloader=test_loader
    )

    accuracy_train = accuracy(embedding_network, classification_network, train_loader, criterion_test)
    accuracy_test = accuracy(embedding_network, classification_network, test_loader, criterion_test)
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


def plot_cifar_weights(network):
    m = [m for m in network.modules() if isinstance(m, nn.Conv2d)][0]
    p = m._parameters['weight'].data
    p = p.view(16, 3, 5, 5)
    print("Dimensions of weights to be plotted:", p.size())
    p = make_grid(p, normalize=True, padding=1)
    npimg = p.cpu().numpy()
    print(npimg.shape)

    plt.figure(figsize=(10,8))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.gca().axis('off')
    plt.show()


plot_cifar_weights(network)
