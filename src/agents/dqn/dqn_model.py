import torch
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256, num_hidden_layers=2):
        super(QNetwork, self).__init__()
        if num_hidden_layers < 1:
            raise ValueError("num_hidden_layers must be >= 1")

        layers = []
        in_features = state_size

        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU())
            in_features = hidden_size

        layers.append(nn.Linear(in_features, action_size))
        self.network = nn.Sequential(*layers)

    def forward(self, state):
        return self.network(state)