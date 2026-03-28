import os
import argparse
import sys
import yaml
import imageio
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environment.shared_core_config import make_env
from agent.dqn_agent import DQNAgent
from agent.dqn_model import QNetwork

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def train(config):
    exp_name = config['experiment_name']
    episodes = config['training']['episodes']
    
    print(f"Starting training for {episodes} episodes. Experiment: {exp_name}")
    env = make_env(render_mode=None) 
    
    state_size = 50 
    action_size = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    agent_params = {**config['agent'], 'hidden_size': config['model']['hidden_size']}
    agent = DQNAgent(state_size, action_size, device, agent_params)
    episode_rewards = []

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    for e in range(episodes):
        state, info = env.reset()
        total_reward = 0
        done = truncated = False
        
        while not (done or truncated):
            action = agent.select_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            agent.step(state, action, reward, next_state, done or truncated)
            state = next_state
            total_reward += reward

        episode_rewards.append(total_reward)
        
        if (e + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {e+1}/{episodes} | Epsilon: {agent.epsilon:.2f} | Avg Reward: {avg_reward:.2f}")

        if (e + 1) % 100 == 0:
            torch.save(agent.policy_net.state_dict(), f"models/{exp_name}_ep{e+1}.pth")

    np.save(f"results/{exp_name}_rewards.npy", np.array(episode_rewards))
    print(f"Training complete. Rewards saved to results/{exp_name}_rewards.npy")
    env.close()

def evaluate(config, target_episode):
    exp_name = config['experiment_name']
    model_path = f"models/{exp_name}_ep{target_episode}.pth"
    
    print(f"Loading model from {model_path}...")
    env = make_env(render_mode="rgb_array")
    
    state_size = 50
    action_size = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = QNetwork(state_size, action_size, config['model']['hidden_size']).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() 
    
    state, info = env.reset(seed=42)
    done = truncated = False
    total_reward = 0
    step = 0
    frames = []
    
    with torch.no_grad(): 
        while not (done or truncated):
            frames.append(env.render())
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            q_values = model(state_tensor)
            action = np.argmax(q_values.cpu().data.numpy())
            state, reward, done, truncated, info = env.step(action)
            total_reward += reward
            step += 1
            
    final_frame = env.render()
    for _ in range(15):
        frames.append(final_frame)

    print(f"Evaluation finished after {step} steps. Total reward: {total_reward:.2f}")
    env.close()

    gif_path = f"results/{exp_name}_rollout.gif"
    imageio.mimsave(gif_path, frames, fps=15)
    print(f"GIF saved successfully to {gif_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Custom DQN Agent")
    parser.add_argument("--mode", type=str, choices=["train", "eval"], required=True)
    parser.add_argument("--config", type=str, default="configs/dqn_params.yaml")
    parser.add_argument("--eval_ep", type=int, default=2000, help="Episode checkpoint to load for eval")
    
    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.mode == "train":
        train(cfg)
    elif args.mode == "eval":
        evaluate(cfg, args.eval_ep)