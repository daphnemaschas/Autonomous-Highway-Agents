import numpy as np

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
    
    for i in range(num_episodes):
        obs, info = env.reset(seed=seed_offset + i)
        done, truncated = False, False
        total_reward = 0.0
        steps = 0
        
        while not (done or truncated):
            action = policy_func(obs)
            
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
    
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    
    return episode_rewards, episode_lengths, mean_reward, std_reward