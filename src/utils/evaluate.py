import numpy as np
from tqdm import tqdm


def evaluate_policy_on_seeds(env, policy_func, eval_seeds):
    """
    Evaluates a policy on an explicit list of seeds.

    Args:
        env (gym.Env): Configured environment.
        policy_func (callable): Function mapping observation -> action.
        eval_seeds (list[int]): Deterministic evaluation seeds shared across models.

    Returns:
        tuple: (episode_rewards, episode_lengths, mean_reward, std_reward, crash_rate, failure_seeds)
    """
    episode_rewards = []
    episode_lengths = []
    crashes = 0
    failure_seeds = []

    for current_seed in tqdm(eval_seeds, desc="Evaluating policy"):
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
            if info.get("crashed", False):
                crashed = True

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        if crashed:
            crashes += 1
            failure_seeds.append(current_seed)

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    crash_rate = (crashes / len(eval_seeds)) * 100
    return episode_rewards, episode_lengths, mean_reward, std_reward, crash_rate, failure_seeds


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
    eval_seeds = [seed_offset + i for i in range(num_episodes)]
    return evaluate_policy_on_seeds(env=env, policy_func=policy_func, eval_seeds=eval_seeds)
