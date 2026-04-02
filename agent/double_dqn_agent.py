import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

from agent.dqn_model import QNetwork
from agent.replay_buffer import ReplayBuffer

class DoubleDQNAgent:
    def __init__(self, state_size, action_size, device, config):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        
        self.gamma = config['gamma']
        self.batch_size = config['batch_size']
        self.target_update_freq = config['target_update_freq']
        self.step_count = 0
        
        self.epsilon = config['epsilon_start']
        self.epsilon_min = config['epsilon_min']
        self.epsilon_decay = config['epsilon_decay']
        
        hidden_size = config.get('hidden_size', 256)
        self.policy_net = QNetwork(state_size, action_size, hidden_size).to(device)
        self.target_net = QNetwork(state_size, action_size, hidden_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config['lr'])
        self.memory = ReplayBuffer(config['buffer_capacity'], device)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        self.policy_net.eval()
        with torch.no_grad():
            action_values = self.policy_net(state_tensor)
        self.policy_net.train()
        
        return np.argmax(action_values.cpu().data.numpy())

    def step(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
        self.step_count += 1
        
        if len(self.memory) > self.batch_size:
            self._learn()
            
        if self.step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def _learn(self):
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        current_q_values = self.policy_net(states).gather(1, actions)
        
        best_actions = self.policy_net(next_states).argmax(1).unsqueeze(1).detach()
        next_q_values = self.target_net(next_states).gather(1, best_actions).detach()
        expected_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        loss = nn.functional.smooth_l1_loss(current_q_values, expected_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay