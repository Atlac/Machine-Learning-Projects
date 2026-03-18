# DQN CliffWalking

A Deep Q-Network (DQN) implementation using PyTorch to solve the CliffWalking-v1 environment from Gymnasium.

## Overview

This project implements a DQN agent that learns to navigate the CliffWalking environment, where an agent must walk from start to goal while avoiding falling off a cliff. The DQN uses neural networks to approximate Q-values and experience replay for stable learning.

## Features

- **Deep Q-Network**: Neural network-based Q-value approximation with fully connected layers
- **Experience Replay**: Stores transitions in a replay buffer for improved sample efficiency
- **Epsilon-Greedy Exploration**: Gradually decreases exploration rate during training
- **One-Hot State Encoding**: Converts discrete states to one-hot vectors for neural network input
- **Training Metrics**: Tracks and visualizes episode rewards during training
- **Model Persistence**: Save and load trained models for evaluation

## Project Structure

```
DQN_ML/
├── DQN.py                      # Main DQN implementation and training script
├── DQN2.py                     # Alternative DQN implementation
├── eval.py                     # Evaluation script for trained models
├── metrics.py                  # Metrics tracking utilities
├── dqn_cliffwalking.pth        # Saved trained model (standard)
├── dqn_cliffwalking2.pth       # Saved trained model (alternative)
└── training_metrics.png        # Training performance visualization
```

## Requirements

- Python 3.7+
- PyTorch
- Gymnasium
- NumPy
- tqdm
- matplotlib (for visualizations)

## Installation

```bash
pip install torch gymnasium numpy tqdm matplotlib
```

## Usage

### Training

Run the main training script:

```bash
python DQN.py
```

This will:
- Train the DQN agent for 600 episodes
- Display progress with average rewards every 50 episodes
- Save the trained model to `dqn_cliffwalking.pth`

### Evaluation

Evaluate a trained model:

```bash
python eval.py
```

## Hyperparameters

The default hyperparameters in `DQN.py` are:

- **Episodes**: 600
- **Batch Size**: 64
- **Gamma (Discount Factor)**: 0.99
- **Learning Rate**: 5e-4
- **Epsilon Start**: 1.0
- **Epsilon End**: 0.05
- **Epsilon Decay**: (1.0 - 0.05) / 30000 steps
- **Replay Memory Size**: 10,000
- **Hidden Dimension**: 64

## Network Architecture

The DQN uses a simple feedforward neural network:

```
Input (obs_dim) → Linear(64) → ReLU → Linear(64) → ReLU → Linear(act_dim)
```

## How It Works

1. **State Representation**: Discrete states are converted to one-hot encoded vectors
2. **Action Selection**: Uses epsilon-greedy policy (explore vs exploit)
3. **Experience Storage**: Transitions (s, a, s', r) are stored in replay memory
4. **Learning**: Samples random batches from memory and updates Q-values using Bellman equation
5. **Loss Function**: Mean Squared Error between predicted and target Q-values

## Results

The trained agent learns to navigate the cliff walking task and avoids falling off the cliff. Training metrics are saved as `training_metrics.png`.

## Environment Details

**CliffWalking-v1**:
- Grid world where the agent starts at the bottom-left
- Goal is at the bottom-right
- A cliff spans the entire bottom row between start and goal
- Falling off the cliff gives -100 reward
- Normal steps give -1 reward
- Reaching the goal terminates the episode

## Notes

- The environment is set to `is_slippery=True` for added difficulty
- Episodes are limited to 200 steps to prevent infinite loops
- The network parameters can be extracted/set using `get_flat_params()` and `set_flat_params()`

## Author

Federico

## License

This project is for educational purposes.
