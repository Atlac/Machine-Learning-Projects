# Q-Learning CliffWalking

A traditional tabular Q-Learning implementation for solving the CliffWalking-v1 environment from Gymnasium. This project provides comprehensive training, evaluation, and visualization capabilities with support for both slippery and non-slippery environments.

## Overview

This project implements a Q-Learning agent that learns optimal policies through table-based value iteration. Unlike deep learning approaches, this uses a Q-table to store state-action values directly, making it interpretable and suitable for discrete state spaces.

## Features

- **Tabular Q-Learning**: Classic reinforcement learning with Q-table storage
- **Dual Environment Support**: Train on slippery or non-slippery variants
- **Two Update Methods**:
  - Standard Q-learning with learning rate
  - Deterministic updates for non-slippery environments
- **Comprehensive Metrics**: Tracks rewards, episode lengths, and success rates
- **Extensive Visualization**: Generates multiple plots for performance analysis
- **Model Persistence**: Save/load Q-tables using pickle
- **Command-Line Interface**: Easy training and evaluation modes

## Project Structure

```
Q_table_ML/
├── Q_learning.py              # Q-Learning agent implementation
├── main.py                    # Main training/evaluation script
├── charts.py                  # Plotting and visualization utilities
├── qtable_slippery.pkl        # Trained Q-table (slippery environment)
├── qtable_not_slippery.pkl    # Trained Q-table (non-slippery environment)
├── plots_*/                   # Training plots for different episode counts
│   ├── *_rewards.png
│   ├── *_lengths.png
│   ├── *_success_rate.png
│   ├── *_epsilon.png
│   ├── *_hyperparams.json
│   └── *_summary.csv
└── plots.zip                  # Archived plots
```

## Requirements

- Python 3.7+
- Gymnasium
- NumPy
- tqdm
- matplotlib
- pickle

## Installation

```bash
pip install gymnasium numpy tqdm matplotlib
```

## Usage

### Training

Train on a **non-slippery** environment (deterministic transitions):

```bash
python main.py --mode train --episodes 2000
```

Train on a **slippery** environment (stochastic transitions):

```bash
python main.py --mode train --episodes 2000 --slippery
```

The script will:
- Train the agent for the specified number of episodes
- Save the Q-table (`qtable_slippery.pkl` or `qtable_not_slippery.pkl`)
- Run 500 evaluation episodes
- Generate comprehensive plots and metrics in `plots_<episodes>/` directory

### Evaluation

Evaluate a trained agent with visual rendering:

```bash
python main.py --mode eval --slippery
```

Or for non-slippery:

```bash
python main.py --mode eval
```

This will:
- Load the corresponding Q-table
- Run 5 episodes with human rendering
- Display performance statistics

## Hyperparameters

Default hyperparameters in `Q_learning.py`:

- **Learning Rate (α)**: 0.0001
- **Discount Factor (γ)**: 0.97
- **Initial Epsilon (ε)**: 1.0
- **Minimum Epsilon**: 0.001
- **Epsilon Decay**: ε / (n_episodes * 0.8)
- **Default Episodes**: 200,000

## Algorithm Details

### Standard Q-Learning Update (Slippery Environment)

```python
Q(s,a) ← (1-α)·Q(s,a) + α·[r + γ·max(Q(s',a'))]
```

### Deterministic Update (Non-Slippery Environment)

```python
Q(s,a) ← r + γ·max(Q(s',a'))
```

## Generated Metrics

For each training run, the following are saved:

1. **Plots** (`plots_<episodes>/`):
   - Episode rewards over time
   - Episode lengths over time
   - Success rate (rolling average)
   - Epsilon decay curve

2. **Hyperparameters** (`*_hyperparams.json`):
   - All training parameters
   - Environment configuration

3. **Evaluation Summary** (`eval_*_summary.csv`):
   - Average reward
   - Maximum reward
   - Minimum reward

## Environment Details

**CliffWalking-v1**:
- **State Space**: 4×12 grid (48 states)
- **Action Space**: 4 actions (Up, Down, Left, Right)
- **Rewards**:
  - Normal step: -1
  - Falling off cliff: -100 (returns to start)
  - Reaching goal: 0 (episode ends)
- **Slippery Mode**: Adds stochasticity to transitions

## Key Classes and Methods

### QLearningAgent

```python
agent = QLearningAgent(env, learning_rate=0.0001, discount_factor=0.97,
                       epsilon=1.0, min_epsilon=0.001, n_episodes=200000)

# Choose action (epsilon-greedy)
action = agent.choose_action(state, greedy=False)

# Update Q-values (standard)
agent.update(state, env)

# Update Q-values (deterministic)
agent.update_qvalue(state, env)

# Save/load Q-table
agent.save_qtable('qtable.pkl')
agent.load_qtable('qtable.pkl')
```

## Training Tips

1. **Non-Slippery Environment**: Use deterministic updates for faster convergence
2. **Slippery Environment**: Requires more episodes and uses standard Q-learning
3. **Episode Count**:
   - Start with 1,000-2,000 episodes for quick testing
   - Use 10,000-50,000 for good performance
   - Use 100,000+ for near-optimal policies

## Results

The Q-Learning agent successfully learns to:
- Navigate from start to goal efficiently
- Avoid the cliff in most episodes
- Achieve near-optimal paths after sufficient training

Success rates and convergence speed vary between slippery and non-slippery environments.

## Comparison with DQN

| Feature | Q-Learning (This Project) | DQN |
|---------|--------------------------|-----|
| State Space | Small discrete spaces | Large/continuous spaces |
| Memory | Q-table (dictionary) | Neural network weights |
| Interpretability | High (inspect Q-values) | Low (black box) |
| Training Speed | Fast for small spaces | Slower, more samples needed |
| Scalability | Poor for large spaces | Excellent |

## Author

Federico

## License

This project is for educational purposes.
