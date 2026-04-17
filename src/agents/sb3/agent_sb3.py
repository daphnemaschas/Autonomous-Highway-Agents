from stable_baselines3 import DQN

def create_sb3_agent(env, log_dir):
    """
    Instantiates and returns a Stable-Baselines3 DQN agent.
    
    Args:
        env (gym.Env): The highway-v0 environment.
        log_dir (str): Directory where TensorBoard logs will be saved.
        
    Returns:
        stable_baselines3.DQN: The initialized DQN model.
    """
    model = DQN(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=5e-4,
        buffer_size=15000,
        learning_starts=200,
        batch_size=32,
        gamma=0.8,
        train_freq=1,
        gradient_steps=1,
        target_update_interval=50,
        exploration_fraction=0.8,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05
    )
    return model