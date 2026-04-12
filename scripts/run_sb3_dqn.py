import argparse
import sys
import os
import numpy as np
import imageio

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agents.sb3.train_sb3 import train
from environment.shared_core_config import make_env
from stable_baselines3 import DQN
from src.utils.evaluate import evaluate_policy

def run_evaluation(model_path, use_safety_wrapper=False, penalty_weight=0.5):
    print(f"Loading model from {model_path}...")
    model = DQN.load(model_path)
    
    env = make_env(use_safety_wrapper=use_safety_wrapper, penalty_weight=penalty_weight)
    
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

    print("\nRecording a long game for the GIF...")
    env_render = make_env(render_mode="rgb_array")
    obs, info = env_render.reset(seed=42)
    done, truncated = False, False
    max_steps = 300
    frames = []

    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env_render.step(action)
        
        current_frame = env_render.render()
        frames.append(current_frame)
        
        if done or truncated:
            for _ in range(10):
                frames.append(current_frame)
            
            obs, info = env_render.reset()
    
    exp_name = os.path.basename(model_path).replace(".zip", "")

    base_folder = "sb3_safety" if use_safety_wrapper else "sb3"
    if use_safety_wrapper:
        base_folder += f"penalty_{penalty_weight}"
        
    gif_path = os.path.join("results", base_folder, f"{exp_name}_long_rollout.gif")
    

    
    imageio.mimsave(gif_path, frames, fps=10)
    print(f"Long GIF saved successfully to {gif_path} !")


if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Run the SB3 DQN Agent")
    parser.add_argument("--mode", type=str, choices=["train", "eval", "train_all"], default="train_all")
    parser.add_argument("--timesteps", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=1, help="Seed for train mode")
    parser.add_argument("--model_path", type=str, default="", help="Path to .zip")
    parser.add_argument("--safety", action="store_true", help="Enable the safety wrapper to avoid crashes")
    parser.add_argument("--penalty_weight", type=float, default=0.5, help="Weight of the close-distance penalty")
    
    args = parser.parse_args()

    if args.mode == "train":
        train(total_timesteps=args.timesteps, seed=args.seed, use_safety_wrapper=args.safety, 
            penalty_weight=args.penalty_weight)

    elif args.mode == "train_all":
        for seed in [1, 2, 3]:
            train(total_timesteps=args.timesteps, seed=seed, use_safety_wrapper=args.safety, 
            penalty_weight=args.penalty_weight)

    elif args.mode == "eval":
        if args.model_path == "":
            print("Error: Specify --model_path (e.g., results/sb3/sb3_seed_1_last.zip)")
            sys.exit(1)
        run_evaluation(args.model_path, use_safety_wrapper=args.safety, 
            penalty_weight=args.penalty_weight)