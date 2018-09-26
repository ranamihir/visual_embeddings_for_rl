import torch
import torch.nn as nn

class EmbeddingNetwork(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(EmbeddingNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 48, kernel_size=5)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 128, kernel_size=5)
        self.fc1 = nn.Linear(128*5*5, 64)
        self.fc2 = nn.Linear(64, num_outputs)

    def forward(self, input):
        input = input.view(-1, 3, 64, 64) # reshape input to batch x num_inputs
        output = torch.tanh(self.conv1(input))
        output = self.pool(output)
        output = torch.tanh(self.conv2(output))
        output = self.pool(output)
        output = output.view(-1, 128*5*5)
        output = self.fc1(output)
        output = self.fc2(output)
        return output

