import gymnasium as gym
import highway_env
from gymnasium.wrappers import FlattenObservation
from src.utils.safety_wrappers import SafetyRewardWrapper

SHARED_CORE_ENV_ID = "highway-v0"

SHARED_CORE_CONFIG = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 10,
        "features": ["presence", "x", "y", "vx", "vy"],
        "absolute": False,
        "normalize": True,
        "clip": True,
        "see_behind": True,
        "observe_intentions": False,
    },
    "action": {
        "type": "DiscreteMetaAction",
        "target_speeds": [20, 25, 30],
    },
    "lanes_count": 4,
    "vehicles_count": 45,
    "controlled_vehicles": 1,
    "initial_lane_id": None,
    "duration": 30,
    "ego_spacing": 2,
    "vehicles_density": 1.0,
    "collision_reward": -1.5,
    "right_lane_reward": 0.0,
    "high_speed_reward": 0.7,
    "lane_change_reward": -0.02,
    "reward_speed_range": [22, 30],
    "normalize_reward": True,
    "offroad_terminal": True,
}

def make_env(render_mode=None, use_safety_wrapper=False, penalty_weight=0.5):
    """
    Creates and configures the highway-v0 environment based on the shared core configuration.
    
    Args:
        render_mode (str, optional): 'rgb_array', 'human', etc.
        use_safety_wrapper (bool): Si True, ajoute la pénalité de distance.
        penalty_weight (float): Puissance de la pénalité de rapprochement.
            
    Returns:
        gym.Env: The configured highway-v0 environment ready for interaction.
    """
    env = gym.make(SHARED_CORE_ENV_ID, render_mode=render_mode)
    env.unwrapped.configure(SHARED_CORE_CONFIG)
    
    if use_safety_wrapper:
        env = SafetyRewardWrapper(env, distance_threshold=15.0, penalty_weight=penalty_weight)

    env.reset() 
    env = FlattenObservation(env)

    return env