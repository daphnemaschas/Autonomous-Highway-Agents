import os
import argparse
import sys
import yaml
import imageio
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environment.shared_core_config import make_env
from agent.double_dqn_per_agent import DoubleDQNPERAgent
from agent.dqn_model import QNetwork


def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def train(config):
    exp_name = config['experiment_name']
    episodes = config['training']['episodes']

    print(f"Starting Double DQN + PER training for {episodes} episodes. Experiment: {exp_name}")
    env = make_env(render_mode=None)

    state_size  = 50
    action_size = env.action_space.n
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent_params = {**config['agent'], 'hidden_size': config['model']['hidden_size']}
    agent = DoubleDQNPERAgent(state_size, action_size, device, agent_params)

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
            print(f"Episode {e+1}/{episodes} | Epsilon: {agent.epsilon:.4f} | Avg Reward: {avg_reward:.2f}")

        if (e + 1) % 100 == 0:
            torch.save(agent.policy_net.state_dict(), f"models/{exp_name}_ep{e+1}.pth")

    torch.save(agent.policy_net.state_dict(), f"models/{exp_name}_final.pth")
    np.save(f"results/{exp_name}_rewards.npy", np.array(episode_rewards))
    env.close()


def evaluate(config, model_path, n_runs=50, seed=0):
    exp_name    = config['experiment_name']
    env         = make_env(render_mode=None)
    state_size  = 50
    action_size = env.action_space.n
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = QNetwork(state_size, action_size, config['model']['hidden_size']).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    rewards = []
    for i in range(n_runs):
        state, _ = env.reset(seed=seed + i)
        done = truncated = False
        total = 0.0
        with torch.no_grad():
            while not (done or truncated):
                t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                action = model(t).argmax().item()
                state, reward, done, truncated, _ = env.step(action)
                total += reward
        rewards.append(total)

    mean_r, std_r = np.mean(rewards), np.std(rewards)
    print(f"Eval {n_runs} runs | seed={seed} | mean={mean_r:.3f} | std={std_r:.3f}")
    np.save(f"results/{exp_name}_eval_seed{seed}.npy", np.array(rewards))
    env.close()
    return mean_r, std_r


def record_rollout(config, model_path):
    exp_name    = config['experiment_name']
    env         = make_env(render_mode="rgb_array")
    state_size  = 50
    action_size = env.action_space.n
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = QNetwork(state_size, action_size, config['model']['hidden_size']).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    state, _ = env.reset(seed=42)
    done = truncated = False
    frames = []

    with torch.no_grad():
        while not (done or truncated):
            frames.append(env.render())
            t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            action = model(t).argmax().item()
            state, _, done, truncated, _ = env.step(action)

    for _ in range(15):
        frames.append(frames[-1])

    gif_path = f"results/{exp_name}_rollout.gif"
    imageio.mimsave(gif_path, frames, fps=15)
    print(f"Rollout saved to {gif_path}")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",       type=str, choices=["train", "eval", "record"], required=True)
    parser.add_argument("--config",     type=str, default="configs/double_dqn_per_params.yaml")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--n_runs",     type=int, default=50)
    parser.add_argument("--seed",       type=int, default=0)

    args = parser.parse_args()
    cfg  = load_config(args.config)

    if args.model_path is None:
        args.model_path = f"models/{cfg['experiment_name']}_final.pth"

    if args.mode == "train":
        train(cfg)
    elif args.mode == "eval":
        evaluate(cfg, args.model_path, n_runs=args.n_runs, seed=args.seed)
    elif args.mode == "record":
        record_rollout(cfg, args.model_path)