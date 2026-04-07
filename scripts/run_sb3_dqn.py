import argparse
import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agents.sb3.train_sb3 import train
from environment.shared_core_config import make_env
from stable_baselines3 import DQN
from src.utils.evaluate import evaluate_policy

def run_evaluation(model_path):
    print(f"Loading model from {model_path}...")
    model = DQN.load(model_path)
    
    env = make_env()
    
    def sb3_policy(obs):
        action, _ = model.predict(obs, deterministic=True)
        return action

    print("Starting deep evaluation...")
    episode_rewards, episode_lengths, mean_reward, std_reward, crash_rate, failure_seeds = evaluate_policy(
        env=env,
        policy_func=sb3_policy,
        num_episodes=50
    )
    
    print("\n--- Evaluation Results ---")
    print(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"Crash Rate: {crash_rate:.1f}%")
    if len(failure_seeds) > 0:
        print(f"Failure seeds (crashes occurred here): {failure_seeds[:5]}...")

    env.close()

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Run the SB3 DQN Agent")
    parser.add_argument("--mode", type=str, choices=["train", "eval", "train_all"], default="train_all")
    parser.add_argument("--timesteps", type=int, default=50000)
    parser.add_argument("--model_path", type=str, default="", help="Path to .zip")
    
    args = parser.parse_args()

    if args.mode == "train":
        train(total_timesteps=args.timesteps, seed=1)

    elif args.mode == "train_all":
        for seed in [1, 2, 3]:
            train(total_timesteps=args.timesteps, seed=seed)

    elif args.mode == "eval":
        if args.model_path == "":
            print("Error: Specify --model_path (e.g., results/sb3/sb3_seed_1_last.zip)")
            sys.exit(1)
        run_evaluation(args.model_path)