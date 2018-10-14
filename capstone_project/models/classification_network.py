import torch
import torch.nn as nn


class ClassificationNetwork(nn.Module):
	def __init__(self, num_inputs, hidden_size, num_outputs):
		super(ClassificationNetwork, self).__init__()
		# Fully connected and ReLU layers
		self.fc1 = nn.Linear(num_inputs, hidden_size)
		self.fc2 = nn.Linear(hidden_size, num_outputs)
		self.relu = nn.ReLU(inplace=True)
		self.dropout = nn.Dropout(p=0.5)

		# Initialize weights
		self._init_weights()

	def forward(self, embedding_output1, embedding_output2):
		input = embedding_output1 * embedding_output2
		input = input.view(input.size(0), -1) # Reshape input to batch_size x num_inputs
		output = self.fc1(input)
		output = self.relu(output)
		output = self.dropout(output)
		output = self.fc2(output)
		return output

	def _init_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Linear):
				nn.init.xavier_normal_(m.weight)
				nn.init.uniform_(m.bias)
