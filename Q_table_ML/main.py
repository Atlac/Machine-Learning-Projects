import gymnasium as gym
import Q_learning
import charts
import argparse
import numpy as np
import os
import json
from tqdm import tqdm

def evaluate(env, n_episodes=1000000):
    total_rewards = []
    for episode in tqdm(range(n_episodes)):
        state_cont = env.reset()[0]
        done = False
        total_reward = 0
        has_fallen = -1
        step = 0
        while not done and step<500:
            
            action = Q_learning.choose_action(state_cont, greedy=True)
           
            #print(f"Chosen action: {action}")
            step += 1
            state_cont, reward, terminated, truncated, info = env.step(action)
            if reward == -100:
                print("Fell off the cliff!")
                has_fallen = episode
            done = terminated or truncated
            total_reward += reward
        total_rewards.append(total_reward)
        mean_reward = np.mean(total_rewards)
        #print(f"Episode {episode+1}/{n_episodes} - Total Reward: {total_reward}")
    max_reward = max(total_rewards)
    min_reward = min(total_rewards)
    print(f"Average Reward over {n_episodes} episodes: {mean_reward}")
    print(f"Maximum Reward over {n_episodes} episodes: {max_reward}")
    print(f"Minimum Reward over {n_episodes} episodes: {min_reward}")
    if has_fallen >= 0:
        print(f"Fell off the cliff in {has_fallen} episodes.")
    else:
        print("Never fell off the cliff in any episode.")
    return total_rewards



def trainig(Q_learning ,env, n_episodes=10000, is_slippery=True):
    state = env.reset()[0]
    if is_slippery:
        print("Training on slippery environment...")
        for episode in tqdm(range(n_episodes)):
            state = env.reset()[0]
            Q_learning.update(state, env)
    elif not is_slippery:
        print("Training on non-slippery environment...")
        for episode in tqdm(range(n_episodes)):
            state = env.reset()[0]
            Q_learning.update_qvalue(state, env)

    
if __name__ == "__main__":

    n_episodes = 30
    
    done = False
 
    parser = argparse.ArgumentParser(description='Run training and evaluation')
    parser.add_argument('--mode', choices=['train', 'eval'], required=True, help='Mode: train or eval')
    parser.add_argument('--episodes', type=int, default=10000, help='Number of episodes for training/evaluation')
    parser.add_argument('--slippery', action='store_true')
    args = parser.parse_args()
    n_episodes = args.episodes
    env = gym.make("CliffWalking-v1", render_mode=None, is_slippery=args.slippery)
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)
    env.reset()
    Q_learning = Q_learning.QLearningAgent(env,n_episodes=n_episodes)
    

    if args.mode == 'train':

        if args.slippery:
            trainig(Q_learning, env, n_episodes,is_slippery=args.slippery)
            
            Q_learning.save_qtable('qtable_slippery.pkl')
        else:
            trainig(Q_learning, env, n_episodes,is_slippery=args.slippery)
            Q_learning.save_qtable('qtable_not_slippery.pkl')
        
        print("Training completed. Starting evaluation...")

        total_rewards = evaluate(env, n_episodes=500)
        print("Evaluation completed. Generating charts...")
        prefix = "eval_" + ("slippery_" if args.slippery else "not_slippery_")
        charts.plot_metrics(total_rewards, prefix=prefix, save_dir=f'plots_{n_episodes}')

        # Save evaluation summary (average, max, min) to a file
        os.makedirs(f'plots_{n_episodes}', exist_ok=True)
        if len(total_rewards) > 0:
            mean_reward = float(np.mean(total_rewards))
            max_reward = float(np.max(total_rewards))
            min_reward = float(np.min(total_rewards))
        else:
            mean_reward = max_reward = min_reward = 0.0

        summary_path = os.path.join(f'plots_{n_episodes}', f"{prefix}summary.csv")
        with open(summary_path, 'w') as f:
            f.write('metric,value\n')
            f.write(f'average_reward,{mean_reward}\n')
            f.write(f'max_reward,{max_reward}\n')
            f.write(f'min_reward,{min_reward}\n')

        print(f"Evaluation stats saved to {summary_path}")

        # Plot metrics
        print("Generating charts...")
        rewards = list(env.return_queue)
        lengths = list(env.length_queue)
        epsilon_history = Q_learning.epsilon_history
        prefix = "slippery_" if args.slippery else "not_slippery_"
        charts.plot_metrics(rewards, lengths, epsilon_history=epsilon_history, save_dir=f'plots_{n_episodes}', successes=Q_learning.episode_successes, prefix=prefix)

        # Save training hyperparameters to JSON
        os.makedirs(f'plots_{n_episodes}', exist_ok=True)
        hyperparams = {
            'learning_rate': float(Q_learning.lr),
            'discount_factor': float(Q_learning.gamma),
            'initial_epsilon': float(Q_learning.epsilon_history[0]) if len(Q_learning.epsilon_history) > 0 else float(Q_learning.epsilon),
            'min_epsilon': float(Q_learning.min_epsilon),
            'epsilon_decay': float(Q_learning.epsilon_decay),
            'n_episodes': int(n_episodes),
            'is_slippery': bool(args.slippery)
        }
        hyper_path = os.path.join(f'plots_{n_episodes}', f"{prefix}hyperparams.json")
        with open(hyper_path, 'w') as hf:
            json.dump(hyperparams, hf, indent=2)
        print(f"Training hyperparameters saved to {hyper_path}")


    
    if args.mode == 'eval':

        
        env1 = gym.make("CliffWalking-v1", render_mode='human', is_slippery=args.slippery)
        if args.slippery:
            Q_learning.load_qtable('qtable_slippery.pkl')
        else:

            Q_learning.load_qtable('qtable_not_slippery.pkl')
        total_rewards = evaluate(env1, n_episodes=5)
        mean_reward = float(np.mean(total_rewards))
        max_reward = float(np.max(total_rewards))
        min_reward = float(np.min(total_rewards))
        print(f"Evaluation results - Average Reward: {mean_reward}, Max Reward: {max_reward}, Min Reward: {min_reward}")



    