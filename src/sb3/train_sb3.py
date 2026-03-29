import os
import sys
from stable_baselines3.common.callbacks import CheckpointCallback

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from shared_core_config import make_env
from src.sb3.agent_sb3 import create_sb3_agent

def train_single_run(total_timesteps=50000, run_id="run_1"):
    """
    Trains a single SB3 DQN model.
    
    Args:
        total_timesteps (int): Number of steps to train.
        run_id (str): Identifier for this run (useful for multiple seeds).
    """
    env = make_env()

    
    model_dir = f"models/sb3/{run_id}/"
    log_dir = f"data/logs/sb3/{run_id}/"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    
    model = create_sb3_agent(env, log_dir)

    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000, 
        save_path=model_dir,
        name_prefix=f"dqn_sb3_{run_id}"
    )

    print(f"--- Starting training for {run_id} ({total_timesteps} steps) ---")
    
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        tb_log_name="dqn"
    )

    
    final_path = os.path.join(model_dir, f"dqn_sb3_final_{run_id}")
    model.save(final_path)
    print(f"--- Training {run_id} completed. Model saved to {final_path}.zip ---")

    env.close()

if __name__ == "__main__": 
    train_single_run(total_timesteps=50000, run_id="seed_1")