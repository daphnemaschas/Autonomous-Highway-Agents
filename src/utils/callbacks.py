import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class EpisodeRewardLoggerCallback(BaseCallback):
    """
    This callback ensures that episodic rewards are recorded and saved 
    as NumPy (.npy) files in the results/ folder, just as the custom DQN does.
    """
    def __init__(self, save_path, verbose=1):
        super().__init__(verbose)
        self.save_path = save_path
        self.episode_rewards = []
        self.current_episode_reward = 0.0

    def _on_step(self) -> bool:
        reward = self.locals["rewards"][0] if type(self.locals["rewards"]) in [list, np.ndarray] else self.locals["rewards"]
        self.current_episode_reward += reward

        done = self.locals["dones"][0] if type(self.locals["dones"]) in [list, np.ndarray] else self.locals["dones"]
        
        if done:
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0.0
            
            if len(self.episode_rewards) % 10 == 0 and self.verbose > 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                print(f"Episode {len(self.episode_rewards)} | Avg Reward: {avg_reward:.2f}")

        return True

    def _on_training_end(self) -> None:
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        np.save(self.save_path, np.array(self.episode_rewards, dtype=np.float32))
        print(f"--- Rewards saved to {self.save_path} ---")