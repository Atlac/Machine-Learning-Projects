import torch
import gymnasium as gym
import numpy as np
# Import DQN class from DQN.py
from DQN import DQN

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_agent(env, policy_net, n_episodes=10):
    policy_net.eval() # Set to evaluation mode
    obs_dim = env.observation_space.n
    total_rewards = []
    
    for episode in range(n_episodes):
        state, info = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        steps = 0
        
        while not (done or truncated) and steps < 500:
            with torch.no_grad():
                state_tensor = torch.tensor(state)
                # Ensure correct dimension
                state_tensor = torch.nn.functional.one_hot(state_tensor, num_classes=obs_dim).float().unsqueeze(0).to(device)
                
                # Select action (greedy)
                logits = policy_net(state_tensor)
                action = torch.argmax(logits).item()
                
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            state = next_state
            steps += 1
            
        total_rewards.append(episode_reward)
        print(f"Episode {episode+1}: Reward = {episode_reward}")
        
    return np.mean(total_rewards)

if __name__ == "__main__":
    env_name = "CliffWalking-v1"
    # Use 'human' render mode to visualize the agent's path
    env = gym.make(env_name, render_mode="human", is_slippery=True) 
    
    obs_dim = env.observation_space.n
    act_dim = env.action_space.n
    
    # Load Model
    model_path = "dqn_cliffwalking.pth"
    policy_net = DQN(obs_dim, act_dim).to(device)
    
    try:
        policy_net.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    except FileNotFoundError:
        print(f"Model file {model_path} not found. Please train first.")
        exit(1)
        
    avg_reward = evaluate_agent(env, policy_net, n_episodes=5)
    print(f"Average Reward: {avg_reward}")
    
    env.close()
