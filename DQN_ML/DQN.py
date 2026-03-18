import copy
import torch
import torch.nn as nn
import random
import numpy as np
from collections import namedtuple, deque
import gymnasium as gym
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




class DQN(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
        nn.Linear(obs_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, act_dim)
        )


    def forward(self, x):
        return self.net(x)


    def select_action(self, obs, deterministic=True):
        if isinstance(obs, np.ndarray):
            obs = torch.FloatTensor(obs).to(next(self.parameters()).device)
        
        logits = self.forward(obs)
        
        if deterministic:
            action = torch.argmax(logits).item()
        else:
            probs = torch.softmax(logits, dim=-1)
            action = torch.multinomial(probs, 1).item()
        return action
    
    def get_flat_params(self):
        """Extract all weights as a single 1D tensor."""
        return torch.cat([p.data.view(-1) for p in self.parameters()])
    
    def set_flat_params(self, flat_params):
        """Set weights from a single 1D tensor."""
        idx = 0
        for p in self.parameters():
            num_param = p.numel()
            p.data.copy_(flat_params[idx:idx + num_param].view_as(p.data))
            idx += num_param


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    

n_episodes = 600
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = (EPS_START - EPS_END) / 30000 
TAU = 0.05
LR = 5e-4


if __name__ == "__main__":
    env = gym.make("CliffWalking-v1", render_mode=None, is_slippery=True)
    obs_dim = env.observation_space.n
    act_dim = env.action_space.n
    state, info = env.reset()

    policy_net = DQN(obs_dim, act_dim).to(device)

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=LR)
    memory = ReplayMemory(10000)

    steps_done = 0

    def select_action(state):
        global steps_done
        sample = random.random()
        eps_threshold = max(EPS_END, EPS_START - steps_done * EPS_DECAY)
        steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # Convert state index to one-hot vector
                state_tensor = torch.tensor(state)
                state_tensor = torch.nn.functional.one_hot(state_tensor, num_classes=obs_dim).float().unsqueeze(0).to(device)
                q_values = policy_net(state_tensor)
                return q_values.argmax().item()
        else:
            return random.randrange(act_dim)



    def optimize_model():
        if len(memory) < BATCH_SIZE:
            return
        transitions = memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        # Convert batch of state indices to one-hot vectors
        state_batch = torch.tensor(batch.state, dtype=torch.long)
        state_batch = torch.nn.functional.one_hot(state_batch, num_classes=obs_dim).float().to(device)
        
        action_batch = torch.LongTensor(batch.action).unsqueeze(1).to(device)
        reward_batch = torch.FloatTensor(batch.reward).to(device)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
        
        # Convert non-final next states to one-hot vectors
        non_final_next_states_list = [s for s in batch.next_state if s is not None]
        non_final_next_states = torch.tensor(non_final_next_states_list, dtype=torch.long)
        non_final_next_states = torch.nn.functional.one_hot(non_final_next_states, num_classes=obs_dim).float().to(device)

        state_action_values = policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = policy_net(non_final_next_states).max(1)[0]
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        criterion = nn.MSELoss()
        loss = criterion(state_action_values.squeeze(), expected_state_action_values)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()




    episode_rewards = []
    for episode in tqdm(range(n_episodes)):
        state, info = env.reset()
        total_reward = 0
        done = False
        episode_steps = 0
        while not done and episode_steps < 200: # Limit steps per episode
            action = select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            episode_steps += 1

            if done:
                next_state_value = None
            else:
                next_state_value = next_state

            memory.push(state, action, next_state_value, reward)
            state = next_state

            optimize_model()
        
        episode_rewards.append(total_reward)
        if (episode + 1) % 50 == 0:
            avg_rew = np.mean(episode_rewards[-50:])
            tqdm.write(f"Episode {episode+1}, Avg Reward: {avg_rew:.2f}, Epsilon: {max(EPS_END, EPS_START - steps_done * EPS_DECAY):.3f}")

    env.close()
    torch.save(policy_net.state_dict(), 'dqn_cliffwalking.pth')
    print("Training complete")

