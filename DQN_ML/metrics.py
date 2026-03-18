import numpy as np
import matplotlib.pyplot as plt


class metrics():
    def plot_metrics(rewards, lengths, losses, epsilons):
        # Create a 2x2 grid of plots
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Rewards (with rolling average)
        axs[0, 0].plot(rewards, alpha=0.3, color='blue', label='Raw')
        window = 50
        if len(rewards) >= window:
            avg_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
            axs[0, 0].plot(range(window-1, len(rewards)), avg_rewards, color='red', label=f'{window}-Ep Avg')
        axs[0, 0].set_title("Rewards per Episode")
        axs[0, 0].set_xlabel("Episode")
        axs[0, 0].legend()

        # 2. Episode Lengths
        axs[0, 1].plot(lengths, color='green')
        axs[0, 1].set_title("Episode Length (Steps)")
        axs[0, 1].set_xlabel("Episode")

        # 3. Training Loss
        axs[1, 0].plot(losses, color='orange')
        axs[1, 0].set_title("Average Training Loss")
        axs[1, 0].set_xlabel("Episode")
        axs[1, 0].set_yscale('log') # Log scale is often better for loss

        # 4. Epsilon Decay
        axs[1, 1].plot(epsilons, color='purple')
        axs[1, 1].set_title("Epsilon Decay")
        axs[1, 1].set_xlabel("Episode")

        plt.savefig("training_metrics.png")
        plt.tight_layout()
        plt.show()