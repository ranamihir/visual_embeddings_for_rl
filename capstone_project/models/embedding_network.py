import logging
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.downsample:
            residual = self.downsample(x)
        x += residual
        x = self.relu(x)
        return x


class CNNNetwork(nn.Module):
    def __init__(self, in_dim, in_channels, hidden_size, out_dim, block=ResidualBlock, num_blocks=3, use_pool=False, use_res=False):
        super(CNNNetwork, self).__init__()
        self.in_dim = in_dim
        self.in_channels = in_channels
        self.use_pool = use_pool
        self.use_res = use_res

        # Conv-ReLU layers with batch-norm
        self.conv1 = conv3x3(in_channels, 32)
        self.bn1 = nn.BatchNorm2d(32)

        # Conv-ReLU layers with batch-norm and downsampling
        self.conv2 = conv3x3(32, 64)
        self.bn2 = nn.BatchNorm2d(64)
        self.downsample1 = nn.MaxPool2d(2) if use_pool else conv3x3(64, 64, stride=2)

        # Conv-ReLU layers with batch-norm and downsampling
        self.conv3 = conv3x3(64, 64)
        self.bn3 = nn.BatchNorm2d(64)
        self.downsample2 = nn.MaxPool2d(2) if use_pool else conv3x3(64, 64, stride=2)

        self.relu = nn.ReLU(inplace=True)

        # Residual layers
        if use_res:
            self.residual_layers = self._make_layer(block, 64, 64, num_blocks)

        # Automatically get dimension of FC layer by using dummy input
        fc1_input_size, _ = self._get_fc_input_size()

        # Fully connected layers
        self.fc1 = nn.Linear(fc1_input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, out_dim)

        self._init_weights() # Initialize weights
        self._get_trainable_params() # Print number of trainable parameters

    def forward(self, x):
        # Reshape input to batch_size x in_channels x height x width
        x = x.view(-1, self.in_channels, self.in_dim, self.in_dim)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.downsample1(x)


        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.downsample2(x)

        if self.use_res:
            x = self.residual_layers(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        # Zero centering and l2 normalization
        x = x - x.mean()
        x = x / torch.norm(x, p=2, dim=1, keepdim=True)

        return x

    def _make_layer(self, block, in_channels, out_channels, num_blocks, stride=1, downsample=None):
        if (not downsample) and ((stride != 1) or (in_channels != out_channels)):
            downsample = nn.Sequential(conv3x3(in_channels, out_channels, stride=stride), nn.BatchNorm2d(out_channels))

        layers = [block(in_channels, out_channels, stride, downsample)]
        for i in range(num_blocks-1):
            # For residual blocks, in_channels = out_channels
            layers.append(block(out_channels, out_channels))

        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.uniform_(m.bias)

    def _get_fc_input_size(self):
        '''
        Returns:
            - input size of FC1 layer
            - tuple of shape (num_channels, height, width)
              for final features before FC1 (useful for decoder)
        '''
        layers = nn.Sequential(self.conv1,
                            self.bn1,
                            self.relu,
                            self.conv2,
                            self.bn2,
                            self.relu,
                            self.downsample1,
                            self.conv3,
                            self.bn3,
                            self.relu,
                            self.downsample2
                            )

        if self.use_res:
            layers = nn.Sequential(layers, self.residual_layers)

        with torch.no_grad():
            dummy_input = torch.zeros([1, self.in_channels, self.in_dim, self.in_dim]).float()
            dummy_output = layers(dummy_input)
            fc_size = dummy_output.flatten(0).shape[0]

        logging.info('Input hidden size of FC1 in Embedding Network: {}'.format(fc_size))

        return fc_size, dummy_output.squeeze(0).shape

    def _get_trainable_params(self):
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logging.info('Number of trainable parameters in EmbeddingNetwork: {}'.format(num_params))
        return num_params


class EmbeddingCNNNetwork(nn.Module):
    def __init__(self, in_dim, in_channels, embedding_size, hidden_size, out_dim):
        super(EmbeddingCNNNetwork, self).__init__()
        self.embedding = nn.Embedding(11, embedding_size)
        self.conv1 = nn.Conv2d(in_channels * embedding_size, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.fc1 = nn.Linear((in_dim//4) * (in_dim//4) * 32, hidden_size)
        self.fc2 = nn.Linear(hidden_size, out_dim)
        self.dropout = nn.Dropout(p=0.5)

        self._init_weights() # Initialize weights
        self._get_trainable_params() # Print number of trainable parameters

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute([0, 1, 4, 2, 3]).contiguous().view(x.size(0), -1, x.size(2), x.size(3))
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        # Zero centering and l2 normalization
        x = x - x.mean()
        x = x / torch.norm(x, p=2, dim=1, keepdim=True)

        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.uniform_(m.bias)

    def _get_trainable_params(self):
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logging.info('Number of trainable parameters in EmbeddingNetwork: {}'.format(num_params))
        return num_params


class RelativeNetwork(nn.Module):
    def __init__(self, in_channels, embedding_size, hidden_size, out_dim):
        super(RelativeNetwork, self).__init__()
        self.embedding = nn.Embedding(11, embedding_size)
        self.conv1 = nn.Conv2d(in_channels * embedding_size, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.conv1x1 = nn.Conv2d(64 + 2, 256, kernel_size=1)
        self.fc1 = nn.Linear(256, hidden_size)
        self.fc2 = nn.Linear(hidden_size, out_dim)
        self.dropout = nn.Dropout(p=0.5)

        self._init_weights() # Initialize weights
        self._get_trainable_params() # Print number of trainable parameters

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute([0, 1, 4, 2, 3]).contiguous().view(x.size(0), -1, x.size(2), x.size(3))

        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))

        coords = torch.linspace(0, x.size(-1) - 1, x.size(-1)).to(x.device) / x.size(-1)
        x_coords = coords.unsqueeze(0).unsqueeze(0).unsqueeze(-1).repeat(x.size(0) ,1, 1, x.size(-1))
        y_coords = coords.unsqueeze(0).unsqueeze(0).unsqueeze(-2).repeat(x.size(0), 1, x.size(-1), 1)
        xy_coord = torch.cat([x_coords, y_coords], 1) # Creates a 2D grid
        xy_coord = xy_coord.view(xy_coord.size(0), xy_coord.size(1), -1) # 2D->1D

        offsets = xy_coord.unsqueeze(-1) - xy_coord.unsqueeze(-2) # Creates offsets for all pixels
        x = x.view(x.size(0), x.size(1), -1) # 2D->1D

        x_ = x.unsqueeze(-1).repeat(1, 1, 1, x.size(-1))
        _x = x.unsqueeze(-2).repeat(1, 1, x.size(-1), 1)
        x = torch.cat([offsets, x_, _x], 1) # Concatenate every pixel with every pixel

        x = F.relu(self.conv1x1(x)) # Do 1x1 convolution to combine information from all pairs

        x = F.avg_pool2d(x, x.size(-1), x.size(-1)).squeeze(-1).squeeze(-1) # Global aggregation of this information
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        # Zero centering and l2 normalization
        x = x - x.mean()
        x = x / torch.norm(x, p=2, dim=1, keepdim=True)

        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.uniform_(m.bias)

    def _get_trainable_params(self):
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logging.info('Number of trainable parameters in EmbeddingNetwork: {}'.format(num_params))
        return num_params
