import logging
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
		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample

	def forward(self, input):
		residual = input
		output = self.conv1(input)
		output = self.bn1(output)
		output = self.relu(output)
		output = self.conv2(output)
		output = self.bn2(output)
		if self.downsample:
			residual = self.downsample(input)
		output += residual
		output = self.relu(output)
		return output


class EmbeddingNetwork(nn.Module):
	def __init__(self, in_dim, in_channels, hidden_size, out_dim, block=ResidualBlock, num_blocks=3, use_pool=False, use_res=False):
		super(EmbeddingNetwork, self).__init__()
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

	def forward(self, input):
		# Reshape input to batch_size x in_channels x height x width
		input = input.view(-1, self.in_channels, self.in_dim, self.in_dim)

		output = self.conv1(input)
		output = self.bn1(output)
		output = self.relu(output)

		output = self.conv2(output)
		output = self.bn2(output)
		output = self.relu(output)
		output = self.downsample1(output)


		output = self.conv3(output)
		output = self.bn3(output)
		output = self.relu(output)
		output = self.downsample2(output)

		if self.use_res:
			output = self.residual_layers(output)

		output = output.view(output.size(0), -1)
		output = self.fc1(output)
		output = self.relu(output)
		output = self.fc2(output)

		# Zero centering and l2 normalization
		output = output - output.mean()
		output = output / torch.norm(output, p=2, dim=1, keepdim=True)

		return output

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
