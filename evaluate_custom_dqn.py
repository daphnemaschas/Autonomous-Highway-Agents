import torch
import numpy as np
from environment.shared_core_config import make_env
from agent.dqn_model import QNetwork

def evaluate():
    env = make_env(render_mode="human")
    
    state_size = 50
    action_size = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = QNetwork(state_size, action_size).to(device)
    
    model_path = "models/dqn_checkpoint_ep500.pth" 
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() 
    
    state, info = env.reset(seed=42)
    done = False
    truncated = False
    total_reward = 0
    step = 0
    
    print(f"Loading model from {model_path} and starting evaluation...")
    
    with torch.no_grad(): 
        while not (done or truncated):
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            
            q_values = model(state_tensor)
            action = np.argmax(q_values.cpu().data.numpy())
            
            state, reward, done, truncated, info = env.step(action)
            total_reward += reward
            step += 1
            
    print(f"Evaluation finished after {step} steps. Total reward: {total_reward:.2f}")
    env.close()

if __name__ == "__main__":
    evaluate()