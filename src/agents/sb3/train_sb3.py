import os
import sys
from stable_baselines3.common.callbacks import CheckpointCallback

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from environment.shared_core_config import make_env
from src.agents.sb3.agent_sb3 import create_sb3_agent

from src.utils.callbacks import EpisodeRewardLoggerCallback

def train(total_timesteps=50000, seed=1, use_safety_wrapper=False, penalty_weight=0.5):
    """
    Trains a single SB3 DQN model.
    
    Args:
        total_timesteps (int): Number of steps to train.
        run_id (str): Identifier for this run (useful for multiple seeds).
    """

    run_id=f"sb3_seed_{seed}"
    base_folder = "sb3_safety" if use_safety_wrapper else "sb3"
    
    env = make_env(use_safety_wrapper=use_safety_wrapper, penalty_weight=penalty_weight)

    model_dir = f"models/{base_folder}/{run_id}/"
    log_dir = f"data/logs/{base_folder}/{run_id}/"
    results_dir = f"results/{base_folder}/"
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    
    model = create_sb3_agent(env, log_dir)
    model.set_random_seed(seed)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000, 
        save_path=model_dir,
        name_prefix=f"dqn_sb3_{run_id}"
    )

    rewards_path = os.path.join(results_dir, f"{run_id}_rewards.npy")
    reward_logger = EpisodeRewardLoggerCallback(save_path=rewards_path)

    print(f"--- Starting training for {run_id} ({total_timesteps} steps) ---")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, reward_logger],
        tb_log_name="dqn"
    )

    
    final_path = os.path.join(results_dir, f"{run_id}_last")
    model.save(final_path)
    print(f"--- Training {run_id} completed. Model saved to {final_path}.zip ---")

    env.close()

if __name__ == "__main__": 
    train(total_timesteps=50000, seed=1)