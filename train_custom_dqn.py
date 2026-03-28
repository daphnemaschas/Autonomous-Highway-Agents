import os
import torch
import numpy as np
from environment.shared_core_config import make_env
from agent.dqn_agent import DQNAgent

def train():
    env = make_env(render_mode=None) 
    
    state_size = 50 
    action_size = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Training on device: {device}")

    agent = DQNAgent(state_size, action_size, device)
    
    episodes = 500
    episode_rewards = []

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    for e in range(episodes):
        state, info = env.reset()
        total_reward = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            action = agent.select_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            
            agent.step(state, action, reward, next_state, done or truncated)
            
            state = next_state
            total_reward += reward

        episode_rewards.append(total_reward)
        
        if (e + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {e+1}/{episodes} | Epsilon: {agent.epsilon:.2f} | Avg Reward (last 10): {avg_reward:.2f}")

        if (e + 1) % 100 == 0:
            torch.save(agent.policy_net.state_dict(), f"models/dqn_checkpoint_ep{e+1}.pth")

    np.save("results/dqn_training_rewards.npy", np.array(episode_rewards))
    print("Training complete. Rewards saved to results/dqn_training_rewards.npy")
    
    env.close()

if __name__ == "__main__":
    train()