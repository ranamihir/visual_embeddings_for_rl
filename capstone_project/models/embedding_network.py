import torch
import torch.nn as nn

#conv3x3
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False)


# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out



class EmbeddingNetwork(nn.Module):
    def __init__(self, in_channels, out_channels,stride=1):
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

