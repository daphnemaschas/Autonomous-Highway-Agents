import random
import torch
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.memory = deque(maxlen=capacity)
        self.device = device

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.memory, batch_size)
        
        states, actions, rewards, next_states, dones = zip(*transitions)

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)