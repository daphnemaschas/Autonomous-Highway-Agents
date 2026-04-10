import gymnasium as gym
import numpy as np

class SafetyRewardWrapper(gym.Wrapper):
    def __init__(self, env, distance_threshold=15.0, penalty_weight=0.5):
        super().__init__(env)
        self.distance_threshold = distance_threshold
        self.penalty_weight = penalty_weight

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        safety_penalty = 0.0
        
        ego_vehicle = self.env.unwrapped.vehicle
        
        for v in self.env.unwrapped.road.vehicles:
            if v is not ego_vehicle:
                distance = np.linalg.norm(ego_vehicle.position - v.position)
                
                if distance < self.distance_threshold:
                    proximity_factor = 1.0 - (distance / self.distance_threshold)
                    safety_penalty -= self.penalty_weight * proximity_factor
                    
        if info.get('crashed', False):
            safety_penalty -= 5.0  

        shaped_reward = reward + safety_penalty
        
        info['safety_penalty'] = safety_penalty
        
        return obs, shaped_reward, terminated, truncated, info