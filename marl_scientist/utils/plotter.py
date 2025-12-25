import matplotlib.pyplot as plt
import os
import numpy as np

def plot_training_curves(agent_histories, filepath="marl_scientist/plots/progress.png"):
    """
    Plots the reward history of agents.
    agent_histories: dict of {agent_id: [rewards...]}
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    
    for agent_id, rewards in agent_histories.items():
        if len(rewards) > 0:
            plt.plot(rewards, marker='o', label=f"Agent {agent_id}")
            
    plt.title("Meta-RL Scientist Training Progress")
    plt.xlabel("Meta-Step")
    plt.ylabel("Mean Reward")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
    
    # print(f"[Plotter] Saved training plot to {filepath}")
    return filepath
