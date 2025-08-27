import torch.nn as nn


class AdvRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, output_size):
        super().__init__()
        # Simple regressor plus a small discriminator branch
        self.feature = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.regressor = nn.Linear(hidden_size, output_size)
        self.discriminator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, adv=False):
        feat = self.feature(x)
        if adv:
            return self.discriminator(feat)
        return self.regressor(feat)
