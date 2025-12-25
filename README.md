# Meta-RL Scientist

**A Closed-Loop Multi-Agent RL Environment for Automated Algorithm Discovery**

## Overview
Meta-RL Scientist is an experimental research environment where "agents" are themselves RL researchers. Their goal is to discover and improve Reinforcement Learning algorithms by proposing configurations, architectures, and loss functions, and verifying them on standard benchmarks.

Unlike purely language-model-based approaches (like "AI Scientist"), this project focuses on **RL-driven evolution**. Agents use Meta-RL and Causal Discovery to learn the dynamics of "what makes an algorithm good," evolving their understanding over time.

## Key Features

- **Agents as Researchers**: Agents observe experiment results and propose new experiments.
- **The "Lab" Environment**: A meta-environment where actions are experiment configurations (e.g., swappable building blocks like Optimizers, Exploration strategies).
- **Inner-Loop Verification**: Proposed algorithms are immediately trained and evaluated on lightweight benchmarks (CartPole, Acrobot, MountainCar) using **Stable Baselines3**.
- **Causal Discovery**: Agents maintain a causal graph of algorithm dynamics (e.g., "Increased entropy coefficient -> Higher exploration -> Better final return").
- **Novelty Search**: Rewards are based not just on performance, but on the novelty of the proposed configuration in the embedding space.

## Architecture

- **Core Engine**: Stable Baselines3 (SB3) for reliable, fast inner-loop training.
- **Action Space**: Configuration Hot-Swapping (JSON-based selection of hyperparameters and modules).
- **Safety**: "Singularity Monitor" to detect and throttle reward hacking or runaway resource usage.

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/meta-rl-scientist.git

# Install dependencies
pip install -e .
```

## Usage

*Coming Soon*

## Roadmap

- [ ] Core Environment Harness (The "Lab")
- [ ] Researcher Agent Implementation (Recurrent Policy)
- [ ] Integration of Stable Baselines3 Runners
- [ ] Causal Discovery Module
- [ ] Meta-RL Training Loop
