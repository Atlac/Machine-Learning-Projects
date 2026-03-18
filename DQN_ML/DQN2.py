import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from tqdm import tqdm
from metrics import metrics
import argparse

# --- Hyperparameters ---
LEARNING_RATE = 0.001
GAMMA = 0.99          # Discount factor for future rewards
EPSILON_START = 1.0   # 100% random actions initially
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
BATCH_SIZE = 64
MEMORY_SIZE = 10000
TARGET_UPDATE = 10    # Update target network every 10 episodes
NUM_EPISODES = 300    # CliffWalking is simple, 500-600 episodes usually suffice

# --- Device Config ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Q-Network ---
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        # Input is size 48 (one-hot encoded state)
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.output = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.output(x)

# --- Replay Buffer ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (torch.stack(states), torch.tensor(actions), torch.tensor(rewards), 
                torch.stack(next_states), torch.tensor(dones))

    def __len__(self):
        return len(self.buffer)

# --- Utility: One-Hot Encoding ---
def one_hot_encode(state, n_states=48):
    """Converts integer state (e.g., 36) to tensor [0,0,..,1,..,0]"""
    one_hot = np.zeros(n_states, dtype=np.float32)
    one_hot[state] = 1.0
    return torch.tensor(one_hot, device=device)

# --- Main Training Loop ---
def train_dqn():
    env = gym.make("CliffWalking-v1")
    
    n_states = env.observation_space.n # 48
    n_actions = env.action_space.n     # 4
    
    policy_net = DQN(n_states, n_actions).to(device)
    target_net = DQN(n_states, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    
    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    memory = ReplayBuffer(MEMORY_SIZE)
    rewards_episodes =np.zeros(NUM_EPISODES)
    epsilon = EPSILON_START
    list_losses = []
    list_epsilons = []
    episode_lengths = []
    
    print("Training started...")
    
    for episode in tqdm(range(NUM_EPISODES)):
        state, _ = env.reset()
        state_tensor = one_hot_encode(state)
        terminated = False
        truncated = False
        episode_steps = 0
        episode_reward = 0
        current_episode_losses = []


        while not (terminated or truncated) and episode_steps < 200:
            # 1. Action Selection (Epsilon-Greedy)
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = policy_net(state_tensor).argmax().item()
            
            # 2. Step Environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            # Modification: Penalty for falling is -100, step is -1.
            # We can leave reward as is or normalize it if training is unstable.
            
            next_state_tensor = one_hot_encode(next_state)
            
            # 3. Store Experience
            memory.push(state_tensor, action, reward, next_state_tensor, terminated)
            
            state = next_state
            state_tensor = next_state_tensor
            episode_steps += 1
            
            # 4. Train Network (Experience Replay)
            if len(memory) >= BATCH_SIZE:
                states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)
                
                states = states.to(device)
                actions = actions.to(device).unsqueeze(1)
                rewards = rewards.to(device)
                next_states = next_states.to(device)
                dones = dones.to(device)
                
                # Compute Q(s, a)
                current_q_values = policy_net(states).gather(1, actions).squeeze()
                
                # Compute max Q(s', a') from Target Net
                with torch.no_grad():
                    max_next_q_values = target_net(next_states).max(1)[0]
                    # If done, target is just reward
                    expected_q_values = rewards + (GAMMA * max_next_q_values * (1 - dones.float()))
                
                # Loss (MSE or Huber)
                loss = F.mse_loss(current_q_values, expected_q_values)
                current_episode_losses.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Decay Epsilon
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        episode_lengths.append(episode_steps)
        
        if len(current_episode_losses) > 0:
            list_losses.append(np.mean(current_episode_losses))
        else:
            list_losses.append(0)
        
        # Update Target Network
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
            
        rewards_episodes[episode] = episode_reward
    metrics.plot_metrics(rewards_episodes, episode_lengths, list_losses, list_epsilons)
    print("Training finished.")
    env.close()
    
    return policy_net

# --- Test the Trained Agent ---
def test_agent(policy_net):
    env = gym.make("CliffWalking-v1", render_mode="human")
    state, _ = env.reset()
    state_tensor = one_hot_encode(state)
    done = False
    
    print("\nVisualizing trained agent...")
    while not done:
        with torch.no_grad():
            action = policy_net(state_tensor).argmax().item()
        state, reward, terminated, truncated, _ = env.step(action)
        state_tensor = one_hot_encode(state)
        done = terminated or truncated

    env.close()

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='Mode: train or test the DQN agent')
    args = argparser.parse_args()
    
    if args.mode == 'train':
        trained_model = train_dqn()
        torch.save(trained_model.state_dict(), 'dqn_cliffwalking2.pth')
    else:
        trained_model = DQN(48, 4).to(device)
        trained_model.load_state_dict(torch.load('dqn_cliffwalking2.pth', map_location=device))
    test_agent(trained_model)
    
