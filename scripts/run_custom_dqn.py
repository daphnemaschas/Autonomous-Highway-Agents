import os
import argparse
import sys
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environment.shared_core_config import make_env
from agent.dqn_agent import DQNAgent
from agent.dqn_model import QNetwork

def train(episodes):
    print(f"Starting training for {episodes} episodes...")
    env = make_env(render_mode=None) 
    
    state_size = 50 
    action_size = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Training on device: {device}")

    agent = DQNAgent(state_size, action_size, device)
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


def evaluate(model_path):
    print(f"Loading model from {model_path} and starting evaluation...")
    env = make_env(render_mode="human")
    
    state_size = 50
    action_size = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = QNetwork(state_size, action_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() 
    
    state, info = env.reset(seed=42)
    done = False
    truncated = False
    total_reward = 0
    step = 0
    
    with torch.no_grad(): 
        while not (done or truncated):
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            
            q_values = model(state_tensor)
            action = np.argmax(q_values.cpu().data.numpy())
            
            state, reward, done, truncated, info = env.step(action)
            total_reward += reward
            step += 1
            
    print(f"Evaluation finished after {step} steps. Total reward: {total_reward:.2f}")
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Custom DQN Agent")
    parser.add_argument("--mode", type=str, choices=["train", "eval"], required=True, 
                        help="Choose whether to train a new model or evaluate an existing one.")
    parser.add_argument("--episodes", type=int, default=500, 
                        help="Number of episodes to train (default: 500).")
    parser.add_argument("--model_path", type=str, default="models/dqn_checkpoint_ep500.pth", 
                        help="Path to the model weights for evaluation.")
    
    args = parser.parse_args()

    if args.mode == "train":
        train(args.episodes)
    elif args.mode == "eval":
        evaluate(args.model_path)