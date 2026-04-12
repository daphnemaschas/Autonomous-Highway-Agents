import torch
import torch.nn as nn


class DuelingQNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256, num_hidden_layers=2):
        super(DuelingQNetwork, self).__init__()
        if num_hidden_layers < 1:
            raise ValueError("num_hidden_layers must be >= 1")

        feature_layers = []
        in_features = state_size
        for _ in range(num_hidden_layers):
            feature_layers.append(nn.Linear(in_features, hidden_size))
            feature_layers.append(nn.ReLU())
            in_features = hidden_size

        self.feature_extractor = nn.Sequential(*feature_layers)
        self.value_stream = nn.Linear(hidden_size, 1)
        self.advantage_stream = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        features = self.feature_extractor(state)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        # Combine value and centered advantage to produce stable Q estimates.
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values
