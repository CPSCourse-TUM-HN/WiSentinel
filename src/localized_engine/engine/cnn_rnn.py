import torch.nn as nn


class CNNRNNRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, output_size):
        super().__init__()
        # e.g. a simple CNN front-end
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        # then an RNN back-end
        self.rnn = nn.LSTM(
            16,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (batch, features) → reshape to (batch, 1, features)
        out = x.unsqueeze(1)
        out = self.conv(out)  # (batch, 16, features/2)
        out = out.transpose(1, 2)  # (batch, seq, channels)
        rnn_out, _ = self.rnn(out)  # (batch, seq, hidden)
        last = rnn_out[:, -1, :]  # (batch, hidden)
        return self.fc(last)  # (batch, output)
