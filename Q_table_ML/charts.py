import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def plot_metrics(rewards, lengths=None, epsilon_history=None, successes=None, save_dir='plots', prefix=''):
    """
    Plots the training metrics: Rewards per episode, Episode lengths, and optionally Epsilon.
    `successes` should be a list of 0/1 per episode; we plot cumulative successes over time.
    """
    os.makedirs(save_dir, exist_ok=True)
    sns.set_theme(style="darkgrid")
    
    # Determine how many subplots we need
    n_plots = 0
    if rewards is not None and len(rewards) > 0: n_plots += 1
    if lengths is not None and len(lengths) > 0: n_plots += 1
    if epsilon_history is not None and len(epsilon_history) > 0: n_plots += 1
    if successes is not None and len(successes) > 0: n_plots += 1
    if n_plots == 0:
        print("No metrics to plot.")
        return

    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 5 * n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]
    elif not isinstance(axes, (list, np.ndarray)):
        axes = [axes]
    
    curr_ax_idx = 0
    
    # Plot Rewards
    if rewards is not None and len(rewards) > 0:
        ax = axes[curr_ax_idx]
        window_size = max(1, len(rewards) // 100)
        
        sns.lineplot(data=rewards, ax=ax, alpha=0.3, label='Episode Reward', color='blue')
        if len(rewards) >= window_size:
            rolling_mean_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
            sns.lineplot(x=np.arange(len(rolling_mean_rewards)) + window_size - 1, y=rolling_mean_rewards, ax=ax, label=f'Rolling Mean ({window_size})', color='darkblue')
        
        ax.set_ylabel('Total Reward')
        ax.set_title('Episode Rewards over Time')
        ax.legend()
        curr_ax_idx += 1
    
    # Plot Lengths
    if lengths is not None and len(lengths) > 0:
        ax = axes[curr_ax_idx]
        window_size = max(1, len(lengths) // 100)
        sns.lineplot(data=lengths, ax=ax, alpha=0.3, label='Episode Length', color='orange')
        if len(lengths) >= window_size:
            rolling_mean_lengths = np.convolve(lengths, np.ones(window_size)/window_size, mode='valid')
            sns.lineplot(x=np.arange(len(rolling_mean_lengths)) + window_size - 1, y=rolling_mean_lengths, ax=ax, label=f'Rolling Mean ({window_size})', color='darkorange')
            
        ax.set_ylabel('Steps')
        ax.set_title('Episode Lengths over Time')
        ax.legend()
        curr_ax_idx += 1
    
    # (TD error plotting removed) -- plots now include Rewards, Lengths, and Epsilon only.
        
    # Plot Epsilon
    if epsilon_history is not None and len(epsilon_history) > 0:
        ax = axes[curr_ax_idx]
        sns.lineplot(data=epsilon_history, ax=ax, color='green')
        ax.set_ylabel('Epsilon')
        ax.set_title('Epsilon Decay')
        ax.set_xlabel('Episode')
        curr_ax_idx += 1
    
    # Plot Success Rate
    if successes is not None and len(successes) > 0:
        ax = axes[curr_ax_idx]
        window_size = 10
        
        # Convert successes to cumulative success rate
        cumulative_successes = np.cumsum(successes)
        cumulative_rate = cumulative_successes / (np.arange(len(successes)) + 1)
        
        sns.lineplot(x=np.arange(len(cumulative_rate)), y=cumulative_rate, ax=ax, alpha=0.5, label='Cumulative Success Rate', color='purple')
        
        # Calculate rolling success rate
        if len(successes) >= window_size:
            rolling_success_rate = np.convolve(successes, np.ones(window_size)/window_size, mode='valid')
            sns.lineplot(x=np.arange(len(rolling_success_rate)) + window_size - 1, y=rolling_success_rate, ax=ax, label=f'Rolling Success Rate ({window_size})', color='darkviolet', linewidth=2)
        
        ax.set_ylabel('Success Rate')
        ax.set_title('Success Rate over Time')
        ax.set_ylim(0, 1.05)  # Success rate is between 0 and 1
        ax.legend()
        curr_ax_idx += 1
    
    # Common X label for the last plot
    if n_plots > 0:
        axes[-1].set_xlabel('Episode')
    
    plt.tight_layout()
    filename = f"{prefix}metrics.png" if prefix else "metrics.png"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path)
    print(f"Metrics plot saved to {save_path}")
    plt.close()
