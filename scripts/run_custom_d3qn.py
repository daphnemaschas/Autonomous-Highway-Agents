import os
import argparse
import sys
import glob
import yaml
import imageio
import torch
import numpy as np
import webbrowser

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environment.shared_core_config import make_env
from src.agents.d3qn.d3qn_agent import D3QNAgent
from src.agents.d3qn.d3qn_model import DuelingQNetwork


def get_latest_trained_model(results_dir="results/d3qn"):
    pattern = os.path.join(results_dir, "*_last.pth")
    candidates = glob.glob(pattern)
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected: true/false")


def train(config):
    exp_name = config["experiment_name"]
    episodes = config["training"]["episodes"]

    print(f"Starting training for {episodes} episodes. Experiment: {exp_name}")
    env = make_env(render_mode=None)

    state_size = 50
    action_size = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent_params = {
        **config["agent"],
        "hidden_size": config["model"]["hidden_size"],
        "num_hidden_layers": config["model"].get("num_hidden_layers", 2),
    }
    agent = D3QNAgent(state_size, action_size, device, agent_params)
    episode_rewards = []

    os.makedirs("models/d3qn", exist_ok=True)
    os.makedirs("results/d3qn", exist_ok=True)

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
            print(
                f"Episode {e+1}/{episodes} | Epsilon: {agent.epsilon:.2f} | Avg Reward: {avg_reward:.2f}"
            )

        if (e + 1) % 100 == 0:
            torch.save(agent.policy_net.state_dict(), f"models/d3qn/{exp_name}_ep{e+1}.pth")

    last_model_path = f"results/d3qn/{exp_name}_last.pth"
    torch.save(agent.policy_net.state_dict(), last_model_path)

    final_rewards_path = f"results/d3qn/{exp_name}_rewards.npy"
    np.save(final_rewards_path, np.array(episode_rewards, dtype=np.float32))

    print(f"Training complete. Last model saved to {last_model_path}")
    print(f"Rewards saved to {final_rewards_path}")
    env.close()


def evaluate(config, model_path, save_and_show_gif=True):
    exp_name = config["experiment_name"]

    print(f"Loading model from {model_path}...")
    env = make_env(render_mode="rgb_array")

    state_size = 50
    action_size = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DuelingQNetwork(
        state_size,
        action_size,
        config["model"]["hidden_size"],
        config["model"].get("num_hidden_layers", 2),
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    num_eval_runs = 50
    rewards = []
    best_reward = -float("inf")
    best_run = -1
    best_seed = None

    with torch.no_grad():
        for run_idx in range(num_eval_runs):
            run_seed = 42 + run_idx
            state, info = env.reset(seed=run_seed)
            done = truncated = False
            total_reward = 0
            step = 0

            while not (done or truncated):
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                q_values = model(state_tensor)
                action = np.argmax(q_values.cpu().data.numpy())
                state, reward, done, truncated, info = env.step(action)
                total_reward += reward
                step += 1

            rewards.append(total_reward)
            print(
                f"Eval run {run_idx + 1}/{num_eval_runs} finished after {step} steps. "
                f"Total reward: {total_reward:.2f}"
            )

            if total_reward > best_reward:
                best_reward = total_reward
                best_run = run_idx + 1
                best_seed = run_seed

    mean_reward = float(np.mean(rewards))
    print(f"Evaluation over {num_eval_runs} runs | Mean reward: {mean_reward:.2f}")
    print(f"Best run: {best_run}/{num_eval_runs} | Best reward: {best_reward:.2f}")

    if save_and_show_gif and best_seed is not None:
        print(f"Replaying best run with seed {best_seed} to generate a single GIF...")
        state, info = env.reset(seed=best_seed)
        done = truncated = False
        best_frames = []

        with torch.no_grad():
            while not (done or truncated):
                best_frames.append(env.render())
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                q_values = model(state_tensor)
                action = np.argmax(q_values.cpu().data.numpy())
                state, reward, done, truncated, info = env.step(action)

        final_frame = best_frames[-1]
        for _ in range(15):
            best_frames.append(final_frame)

        gif_path = f"results/d3qn/{exp_name}_best_rollout.gif"
        imageio.mimsave(gif_path, best_frames, fps=15)
        print(f"Best-run GIF saved successfully to {gif_path}")

        try:
            webbrowser.open(f"file://{os.path.abspath(gif_path)}")
            print("Opened GIF with the system default viewer.")
        except Exception as exc:
            print(f"Could not open GIF automatically: {exc}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Custom Dueling Double DQN Agent")
    parser.add_argument("--mode", type=str, choices=["train", "eval"], required=True)
    parser.add_argument("--config", type=str, default="configs/d3qn_params.yaml")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to model weights. Defaults to the latest trained *_last.pth in results/d3qn/.",
    )
    parser.add_argument(
        "--save_and_show_gif",
        type=str2bool,
        default=True,
        help="Whether to save and open the best-run GIF during evaluation (true/false).",
    )

    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.mode == "train":
        train(cfg)
    elif args.mode == "eval":
        if args.model_path is None:
            args.model_path = get_latest_trained_model("results/d3qn")
            if args.model_path is None:
                exp_name = cfg["experiment_name"]
                args.model_path = f"results/d3qn/{exp_name}_last.pth"
                raise FileNotFoundError(
                    f"No trained *_last.pth model found in results/d3qn/. Expected fallback path: {args.model_path}"
                )

        evaluate(cfg, args.model_path, args.save_and_show_gif)
