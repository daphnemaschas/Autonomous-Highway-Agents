import sys
import os
import gymnasium as gym

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from shared_core_config import make_env

def main():
    """
    Test the environment configuration visually with random actions.
    """
    env = make_env(render_mode="human")
    
    obs, info = env.reset(seed=42)
    print("Initial observation shape:", obs.shape)
    print("Action space:", env.action_space)

    done = truncated = False
    step = 0
    total_reward = 0.0

    while not (done or truncated) and step < 100:
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        step += 1

    print(f"Episode finished after {step} steps. Total reward: {total_reward:.2f}")
    env.close()

if __name__ == "__main__":
    main()