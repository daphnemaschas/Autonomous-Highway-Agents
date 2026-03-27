import numpy as np
from tqdm import tqdm

def evaluate_policy(env, policy_func, num_episodes=50, seed_offset=100):
    """
    Evaluates a given policy over a specified number of episodes.
    Requires exactly 50 runs by the assignment "evaluate thoroughly (50 runs mean rewards + std)".
    
    Args:
        env (gym.Env): The configured highway-v0 environment.
        policy_func (callable): A function that takes an observation and returns an action.
                                For SB3, it can be `lambda obs: model.predict(obs, deterministic=True)[0]`.
        num_episodes (int): Number of episodes to run (default to 50).
        seed_offset (int): Offset for environment seeds to ensure evaluation isn't on training seeds.
        
    Returns:
        tuple: (list of rewards, list of episode lengths, mean reward, standard deviation)
    """
    episode_rewards = []
    episode_lengths = []
    crashes = 0
    failure_seeds = []
    
    for i in tqdm(range(num_episodes), desc="Evaluating policy"):
        current_seed = seed_offset + i
        obs, info = env.reset(seed=current_seed)
        done, truncated = False, False
        total_reward = 0.0
        steps = 0
        crashed = False
        
        while not (done or truncated):
            action = policy_func(obs)
            
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            if info.get('crashed', False):
                crashed = True
            
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        if crashed:
            crashes += 1
            failure_seeds.append(current_seed)
    
    
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    crash_rate = (crashes / num_episodes) * 100
    
    return episode_rewards, episode_lengths, mean_reward, std_reward