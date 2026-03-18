
import numpy as np
from collections import defaultdict
from tqdm import tqdm



class QLearningAgent:
    def __init__(self, env, learning_rate=0.0001, discount_factor= 0.97, epsilon=1.0, min_epsilon=0.001, n_episodes=200000):
        self.env = env 
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = self.epsilon/(n_episodes*0.8)
        self.min_epsilon = min_epsilon
        self.q_table = defaultdict(lambda: np.zeros(env.action_space.n))
        
        self.action_space = env.action_space.n
        # Episode metrics (returns and lengths)
        self.episode_returns = []
        self.episode_lengths = []
        # Episode success flags (1 = success, 0 = failure)
        self.episode_successes = []
        self.epsilon_history = []
       
    def choose_action(self, state, greedy=False):
        
        if (not greedy) and (np.random.rand() < self.epsilon):
            
            action = self.env.action_space.sample()
            #print(f"Random action: {action}")
            return action
        
        action = int(np.argmax(self.q_table[state]))
        #print(f"Greedy action: {action}")
        return action


    
    def update_qvalue(self, state, env):
        done = False
        episode_return = 0.0
        episode_length = 0
        fell = False

        while not done:
            action = self.choose_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)

            episode_return += reward
            episode_length += 1
            if reward == -100:
                fell = True

            #deterministic update
            self.q_table[state][action] = (reward + self.gamma * np.max(self.q_table[next_state]) * (not (terminated or truncated)))

            state = next_state
            done = terminated or truncated

        self.decay_epsilon()
        # store episode metrics and success flag
        self.episode_returns.append(float(episode_return))
        self.episode_lengths.append(int(episode_length))
        self.episode_successes.append(int(not fell))


    def update(self, state, env):
        done = False

        episode_return = 0.0
        episode_length = 0
        fell = False
        while not done:
            action = self.choose_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)

            episode_return += reward
            episode_length += 1
            if reward == -100:
                fell = True

            next_max = (not (terminated or truncated)) * np.max(self.q_table[next_state])

            self.q_table[state][action] = (
                (1 - self.lr) * self.q_table[state][action]
                + self.lr * (reward + self.gamma * next_max)
            )

            state = next_state
            done = terminated or truncated

        self.decay_epsilon()
        # store episode metrics
        self.episode_returns.append(float(episode_return))
        self.episode_lengths.append(int(episode_length))
        self.episode_successes.append(int(not fell))
            
       
    def save_qtable(self, filename):
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump(dict(self.q_table), f)
    
    def load_qtable(self, filename):
        import pickle
        with open(filename, 'rb') as f:
            q_table_dict = pickle.load(f)
            self.q_table = defaultdict(lambda: np.zeros(self.env.action_space.n), q_table_dict)


    def decay_epsilon(self):
        self.epsilon_history.append(self.epsilon)
        if self.epsilon > self.min_epsilon:
            self.epsilon -= self.epsilon_decay

