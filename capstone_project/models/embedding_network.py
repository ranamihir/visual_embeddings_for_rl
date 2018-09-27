import torch
import torch.nn as nn


def conv3x3(in_channels, out_channels, stride=1, padding=1, bias=False):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=padding, bias=bias)


# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, input):
        residual = input
        output = self.conv1(input)
        output = self.bn1(output)
        output = torch.Relu(output)
        output = self.conv2(output)
        output = self.bn2(output)
        if self.downsample:
            residual = self.downsample(input)
        output += residual
        output = torch.Relu(output)
        return output


class EmbeddingNetwork(nn.Module):
    def __init__(self, in_dim, in_channels, out_dim, block=ResidualBlock, hidden_size=1024, num_blocks=3):
        super(EmbeddingNetwork, self).__init__()
        self.in_dim = in_dim
        self.in_channels = in_channels
        #conv layers with downsample
        self.conv1 = conv3x3(2, 32)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = conv3x3(32, 64)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = conv3x3(64, 64)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2)

        # Residual layers
        self.residual_layers = self._make_layer(block, 64, 64, num_blocks)

        self.fc1 = nn.Linear(64*16*16, hidden_size)
        self.fc2 = nn.Linear(hidden_size, out_dim)

        self._init_weights()

    def forward(self, input):
        # Reshape input to batch_size x in_channels x height x width
        input = input.view(-1, self.in_channels, self.in_dim, self.in_dim)

        output = self.conv1(input)
        output = self.bn1(output)
        output = torch.Relu(output)

        output = self.conv2(output)
        output = self.bn2(output)
        output = torch.Relu(output)
        output = self.pool(output)

        output = self.conv3(output)
        output = self.bn3(output)
        output = torch.Relu(output)
        output = self.pool(output)

        output = self.residual_layers(output)

        output = output.view(output.size(0), -1)
        output = self.fc1(output)
        output = self.fc2(output)
        return output

    def _make_layer(self, block, in_channels, out_channels, num_blocks, stride=1, downsample=None):
        if not downsample and ((stride != 1) or (self.in_channels != out_channels)):
            downsample = nn.Sequential(conv3x3(self.in_channels, out_channels, stride=stride), nn.BatchNorm2d(out_channels))

        layers = [block(in_channels, out_channels, stride, downsample))]
        for i in range(num_blocks-1):
            # For residual blocks, in_channels = out_channels
            layers.append(block(out_channels, out_channels))

        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m._parameters['weight'])
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m._parameters['weight'], 1)
                nn.init.constant_(m._parameters['bias'], 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m._parameters['weight'])
                nn.init.uniform_(m._parameters['bias'])
