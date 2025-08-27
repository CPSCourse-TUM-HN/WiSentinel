import torch.nn as nn


class SENRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, output_size):
        super().__init__()
        # Example: a Dense SENet-style block
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.gate = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Sigmoid())
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        y = self.relu(self.fc1(x))
        g = self.gate(y)
        y = g * self.fc2(y) + (1 - g) * y
        return self.out(y)
