import torch.nn as nn


class PrimalNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PrimalNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.network(x)


class DualNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DualNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softplus()
        )

    def forward(self, x):
        return self.network(x)
