import torch
import torch.nn as nn

# options
DATASET = 'moving_mnist'

if DATASET == 'moving_mnist':
    num_inputs, n_channels = 64, 1
    num_outputs = 6
elif DATASET == 'cifar10':
    num_inputs, n_channels = 32, 3
    num_outputs = 10

class ClassificationNetwork(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(ClassificationNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 128, kernel_size=5)
        self.fc1 = nn.Linear(128*5*5, 64)
        self.fc2 = nn.Linear(64, num_outputs)

    def forward(self, input):
        input = input.view(-1, 3, 32, 32) # reshape input to batch x num_inputs
        output = torch.tanh(self.conv1(input))
        output = self.pool(output)
        output = torch.tanh(self.conv2(output))
        output = self.pool(output)
        output = output.view(-1, 128*5*5)
        output = self.fc1(output)
        output = self.fc2(output)
        return output

