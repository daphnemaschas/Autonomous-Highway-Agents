import numpy as np
import torch
from agent.sum_tree import SumTree


class PrioritizedReplayBuffer:
    def __init__(self, capacity, device, alpha=0.6, beta=0.4, beta_increment=0.001, eps=1e-6):
        self.tree     = SumTree(capacity)
        self.capacity = capacity
        self.device   = device
        self.alpha    = alpha
        self.beta     = beta
        self.beta_increment = beta_increment
        self.eps      = eps
        self.max_priority = 1.0

    def push(self, state, action, reward, next_state, done):
        self.tree.add(self.max_priority ** self.alpha, (state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch      = []
        tree_idxs  = []
        priorities = []

        segment = self.tree.total() / batch_size

        self.beta = min(1.0, self.beta + self.beta_increment)

        for i in range(batch_size):
            s = np.random.uniform(segment * i, segment * (i + 1))
            idx, priority, data = self.tree.get(s)
            batch.append(data)
            tree_idxs.append(idx)
            priorities.append(priority)

        total    = self.tree.total()
        probs    = np.array(priorities) / total
        weights  = (len(self.tree) * probs) ** (-self.beta)
        weights /= weights.max()  # normalize
        weights  = torch.tensor(weights, dtype=torch.float32).unsqueeze(1).to(self.device)

        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.tensor(np.array(states),      dtype=torch.float32).to(self.device),
            torch.tensor(np.array(actions),     dtype=torch.long).unsqueeze(1).to(self.device),
            torch.tensor(np.array(rewards),     dtype=torch.float32).unsqueeze(1).to(self.device),
            torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device),
            torch.tensor(np.array(dones),       dtype=torch.float32).unsqueeze(1).to(self.device),
            weights,
            tree_idxs,
        )

    def update_priorities(self, tree_idxs, td_errors):
        for idx, error in zip(tree_idxs, td_errors):
            priority = (abs(error) + self.eps) ** self.alpha
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(idx, priority)

    def __len__(self):
        return len(self.tree)