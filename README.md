# Machine Learning Projects

A collection of reinforcement learning projects demonstrating different approaches to solving the CliffWalking environment. This repository compares tabular Q-Learning with Deep Q-Networks (DQN) for discrete control tasks.

## Projects Overview

### 1. Q-Learning (Tabular Method)
Located in `Q_table_ML/`

A traditional Q-Learning implementation using a Q-table to store state-action values. This approach is ideal for small discrete state spaces and provides interpretable results.

**Key Features:**
- Tabular Q-Learning with Q-table storage
- Support for both slippery and non-slippery environments
- Comprehensive metrics and visualization
- Fast training for discrete spaces

[View Q-Learning Project →](Q_table_ML/README.md)

### 2. Deep Q-Network (DQN)
Located in `DQN_ML/`

A neural network-based approach using PyTorch that approximates Q-values with a deep learning model. This method scales to larger state spaces and uses experience replay for stable learning.

**Key Features:**
- Neural network Q-value approximation
- Experience replay buffer
- Epsilon-greedy exploration
- One-hot state encoding

[View DQN Project →](DQN_ML/README.md)

## Environment

Both projects solve the **CliffWalking-v1** environment from Gymnasium:
- **Objective**: Navigate from start (bottom-left) to goal (bottom-right)
- **Challenge**: Avoid falling off the cliff spanning the bottom row
- **State Space**: 4×12 grid (48 states)
- **Action Space**: 4 actions (Up, Down, Left, Right)
- **Rewards**:
  - Normal step: -1
  - Falling off cliff: -100
  - Reaching goal: 0 (terminal)

## Comparison: Q-Learning vs DQN

| Aspect | Q-Learning | DQN |
|--------|------------|-----|
| **State Representation** | Direct table lookup | One-hot encoded vectors |
| **Memory Structure** | Q-table (dictionary) | Neural network + Replay buffer |
| **Scalability** | Limited to small discrete spaces | Handles large/continuous spaces |
| **Training Speed** | Fast for small environments | Slower, requires more samples |
| **Interpretability** | High (inspect Q-values directly) | Low (neural network black box) |
| **Sample Efficiency** | Good for discrete spaces | Lower (needs experience replay) |
| **Generalization** | None (every state learned) | Can generalize to unseen states |
| **Memory Usage** | O(states × actions) | O(network parameters) |

## Quick Start

### Prerequisites

Install common dependencies:
```bash
pip install gymnasium numpy tqdm matplotlib
```

For DQN, additionally install PyTorch:
```bash
pip install torch
```

### Training Q-Learning Agent

```bash
cd Q_table_ML
python main.py --mode train --episodes 2000
```

### Training DQN Agent

```bash
cd DQN_ML
python DQN.py
```

### Evaluation

**Q-Learning:**
```bash
cd Q_table_ML
python main.py --mode eval
```

**DQN:**
```bash
cd DQN_ML
python eval.py
```

## Project Structure

```
Machine Learning Project/
├── README.md                    # This file
├── Q_table_ML/                  # Tabular Q-Learning implementation
│   ├── README.md
│   ├── main.py
│   ├── Q_learning.py
│   ├── charts.py
│   └── qtable_*.pkl             # Trained Q-tables
└── DQN_ML/                      # Deep Q-Network implementation
    ├── README.md
    ├── DQN.py
    ├── eval.py
    ├── metrics.py
    └── dqn_cliffwalking*.pth    # Trained models
```

## When to Use Which Approach?

### Use Q-Learning When:
- State space is small and discrete (< 10,000 states)
- You need interpretable results
- Fast training is important
- You want guaranteed convergence with proper hyperparameters

### Use DQN When:
- State space is large or continuous
- States have natural similarity (can benefit from generalization)
- You're working with visual observations (pixels)
- You need to scale to complex environments

## Results

Both approaches successfully learn to navigate the CliffWalking environment:
- **Q-Learning**: Achieves near-optimal policies with 10,000-50,000 episodes
- **DQN**: Learns effective policies with 600 episodes using neural network approximation

Training metrics and visualizations are saved in each project directory.

## Learning Objectives

This repository demonstrates:
1. **Tabular vs Function Approximation**: Direct Q-tables vs neural networks
2. **Experience Replay**: How replay buffers stabilize deep RL training
3. **Exploration Strategies**: Epsilon-greedy policies in both approaches
4. **Discrete Control**: Solving grid-world navigation tasks
5. **RL Fundamentals**: Bellman equations, TD learning, and policy improvement

## Author

Federico

## License

These projects are for educational purposes.

## Further Reading

- [Sutton & Barto: Reinforcement Learning](http://incompleteideas.net/book/the-book.html)
- [DQN Paper (Mnih et al., 2015)](https://www.nature.com/articles/nature14236)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
