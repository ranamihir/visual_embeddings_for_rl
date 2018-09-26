import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable


def train(embedding_network, classification_network, dataloader, criterion, optimizer, device, epoch):
    embedding_network.train()
    classification_network.train()
    loss_train = 0.
    for batch_idx, (x1, x2, y) in enumerate(dataloader):
        x1, x2, y = Variable(x1).to(device), Variable(x2).to(device), Variable(y).to(device)
        optimizer.zero_grad()
        embedding_output1 = embedding_network(x1)
        embedding_output2 = embedding_network(x2)
        classification_input = torch.dot(embedding_output1, embedding_output2)
        classification_output = classification_network(classification_input)
        loss = criterion(classification_output, y)
        loss.backward()
        optimizer.step()

        # Accurately compute loss, because of different batch size
        loss_train += loss.item() * len(x) / len(dataloader.dataset)

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(x), len(dataloader.dataset),
                100. * batch_idx / len(dataloader), loss.item()))

    optimizer.zero_grad()
    return loss_train

def test(embedding_network, classification_network, dataloader, criterion, device):
    embedding_network.eval()
    classification_network.eval()
    loss_test = 0.
    y_ls = []
    output_ls = []
    with torch.no_grad():
        for batch_idx, (x1, x2, y) in enumerate(dataloader):
            x1, x2, y = Variable(x1).to(device), Variable(x2).to(device), Variable(y).to(device)
            embedding_output1 = embedding_network(x1)
            embedding_output2 = embedding_network(x2)
            classification_input = torch.dot(embedding_output1, embedding_output2)
            classification_output = classification_network(classification_input)
            loss = criterion(classification_output, y)

            # Accurately compute loss, because of different batch size
            loss_test += loss.item() / len(dataloader.dataset)

            output_ls.append(classification_output)
            y_ls.append(y)
    optimizer.zero_grad()
    return loss_test, torch.cat(output_ls, dim=0), torch.cat(y_ls, dim=0)

def accuracy(embedding_network, classification_network, dataloader, criterion, device):
    _, y_predicted, y_true = test(
        embedding_network=embedding_network,
        classification_network=classification_network,
        dataloader=dataloader,
        criterion=criterion,
        device=device
    )

    y_predicted = y_predicted.max(1)[1]
    return 100*y_predicted.eq(y_true.data.view_as(y_predicted)).float().mean().item()

def save_plot(project_dir, fig, filename):
    plot_path = os.path.join(project_dir, 'plots')
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    fig.savefig(os.path.join(plot_path, 'data_sample.png'))
