import re
import matplotlib.pyplot as plt
import os
import numpy as np

LOG_FILE = "simulation.log"
OUTPUT_DIR = "analysis_output"

def parse_log(filepath):
    rewards = {} # aid -> list of (step, reward)
    losses = [] # list of (step, loss)
    
    current_step = 0
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            # Track progress or approximate step
            # "[Progress: 24/1000]"
            prog_match = re.search(r"Progress: (\d+)/", line)
            if prog_match:
                current_step = int(prog_match.group(1))
            
            # Reward
            # [Event] Experiment completed for Agent_1_h0 (Agent: Agent_1). Reward: 50.0
            rew_match = re.search(r"Experiment completed for .* \(Agent: (.*?)\)\. Reward: ([-+]?[0-9]*\.?[0-9]+)", line)
            if rew_match:
                aid = rew_match.group(1)
                rew = float(rew_match.group(2))
                if aid not in rewards:
                    rewards[aid] = []
                rewards[aid].append((current_step, rew))
                
            # Loss
            # - Meta-Loss: 0.0234
            loss_match = re.search(r"Meta-Loss: ([-+]?[0-9]*\.?[0-9]+)", line)
            if loss_match:
                losses.append((current_step, float(loss_match.group(1))))

    return rewards, losses

def plot_rewards(rewards):
    plt.figure(figsize=(12, 6))
    
    all_rewards = []
    
    for aid, data in rewards.items():
        if not data: continue
        steps = [d[0] for d in data]
        rews = [d[1] for d in data]
        all_rewards.extend([d[1] for d in data])
        
        # Smooth
        if len(rews) > 5:
            smooth_rews = np.convolve(rews, np.ones(5)/5, mode='valid')
            plt.plot(steps[2:-2], smooth_rews, label=f"{aid} (smooth)")
            plt.plot(steps, rews, alpha=0.2, linestyle='dotted')
        else:
            plt.plot(steps, rews, alpha=0.5, label=aid, marker='o', markersize=3)
            
    plt.title("Agent Rewards over Time")
    plt.xlabel("Global Completions")
    plt.ylabel("Reward")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "rewards_history.png"))
    print(f"Saved rewards_history.png")
    
    # Best vs Avg
    if all_rewards:
        plt.figure(figsize=(12,4))
        plt.hist(all_rewards, bins=30, alpha=0.7)
        plt.title("Reward Distribution")
        plt.xlabel("Reward")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "rewards_dist.png"))

def plot_loss(losses):
    if not losses:
        print("No loss data found.")
        return
        
    plt.figure(figsize=(10, 5))
    steps = [d[0] for d in losses]
    vals = [d[1] for d in losses]
    
    plt.plot(steps, vals, color='red', marker='x')
    plt.title("Meta-Brain Training Loss")
    plt.xlabel("Global Completions")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "meta_loss.png"))
    print(f"Saved meta_loss.png")

if __name__ == "__main__":
    if not os.path.exists(LOG_FILE):
        print(f"Log file {LOG_FILE} not found.")
        exit()
        
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Parsing logs...")
    rewards, losses = parse_log(LOG_FILE)
    
    print(f"Found traces for {len(rewards)} agents.")
    print(f"Found {len(losses)} loss entries.")
    
    plot_rewards(rewards)
    plot_loss(losses)
