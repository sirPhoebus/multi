import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal
import numpy as np
from typing import List, Dict, Any, Optional
from marl_scientist.core import ExperimentConfig, ExperimentResult
from marl_scientist.agents.brain import MetaBrain, BrainEncoder

class MetaPPOTrainer:
    """
    Standard PPO implementation tailored for the MetaBrain.
    Trains the 'Scientist' Brain.
    """
    def __init__(
        self, 
        brain: MetaBrain, 
        lr=3e-4, 
        gamma=0.99, 
        eps_clip=0.2, 
        k_epochs=4,
        entropy_coef=0.01
    ):
        self.brain = brain
        self.optimizer = optim.Adam(self.brain.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.entropy_coef = entropy_coef
        
        self.mse_loss = nn.MSELoss()
        
    def select_action(self, inputs: Dict[str, torch.Tensor], hidden_state: Optional[torch.Tensor]):
        """
        Samples actions from the brain and returns log_probs for training.
        """
        self.brain.eval()
        with torch.no_grad():
            algo_logits, env_logits, hp_means, value, new_hidden = self.brain(
                inputs["history"],
                inputs["trends"],
                inputs["knowledge"],
                hidden_state
            )
            
            # Algo selection
            dist_algo = Categorical(logits=algo_logits)
            algo_idx = dist_algo.sample()
            
            # Env selection
            dist_env = Categorical(logits=env_logits)
            env_idx = dist_env.sample()
            
            # HP selection (using repo's shared logstd)
            hp_std = torch.exp(self.brain.hp_logstd)
            dist_hp = Normal(hp_means, hp_std)
            hp_vals = dist_hp.sample()
            
            # Pack log probs
            log_prob = dist_algo.log_prob(algo_idx) + \
                       dist_env.log_prob(env_idx) + \
                       dist_hp.log_prob(hp_vals).sum(dim=-1)
                       
        return {
            "algo_idx": algo_idx.item(),
            "env_idx": env_idx.item(),
            "hp_vals": hp_vals.squeeze(0).cpu().numpy(),
            "log_prob": log_prob.item(),
            "value": value.item(),
            "hidden": hidden_state,
            "new_hidden": new_hidden
        }

    def update(self, memory: List[Dict[str, Any]]):
        """
        Performs PPO update on collected trajectories.
        """
        self.brain.train()
        
        # 1. Prepare Tensors
        # Flattened memory to batch
        # We assume single-trajectory updates for now, or multiple if memory is large
        
        # Calculate Returns and Advantages (GAE or simple)
        rewards = [m['reward'] for m in memory]
        values = [m['value'] for m in memory]
        is_terminals = [m['done'] for m in memory]
        
        returns = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(rewards), reversed(is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            returns.insert(0, discounted_reward)
            
        returns = torch.tensor(returns, dtype=torch.float32)
        old_log_probs = torch.tensor([m['log_prob'] for m in memory], dtype=torch.float32)
        old_values = torch.tensor(values, dtype=torch.float32)
        advantages = returns - old_values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        for _ in range(self.k_epochs):
            # Evaluate current policy
            # We need to re-run the brain on ALL steps in memory (preserving hidden states)
            # This is slow but necessary for RNN. 
            # In a real system, we'd use BPTT or fixed hidden states.
            
            total_loss = 0
            
            # To handle RNN properly, we should iterate in sequence or use packed sequences.
            # For this MVP, let's treat each step as independent or use the stored hidden states 
            # (which is 'fixed-state' policy optimization).
            
            for i, m in enumerate(memory):
                inputs = m['inputs']
                h = m['hidden']
                
                curr_algo_logits, curr_env_logits, curr_hp_means, curr_value, _ = self.brain(
                    inputs["history"],
                    inputs["trends"],
                    inputs["knowledge"],
                    h
                )
                
                # Distributions
                d_algo = Categorical(logits=curr_algo_logits)
                d_env = Categorical(logits=curr_env_logits)
                d_hp = Normal(curr_hp_means, torch.exp(self.brain.hp_logstd))
                
                # New log probs of the OLD actions
                lp_algo = d_algo.log_prob(torch.tensor([m['algo_idx']]))
                lp_env = d_env.log_prob(torch.tensor([m['env_idx']]))
                lp_hp = d_hp.log_prob(torch.tensor(m['hp_vals'])).sum(dim=-1)
                
                new_lp = lp_algo + lp_env + lp_hp
                entropy = d_algo.entropy() + d_env.entropy() + d_hp.entropy().sum()
                
                # PPO Ratio
                ratio = torch.exp(new_lp - old_log_probs[i])
                
                # Surrogate Loss
                surr1 = ratio * advantages[i]
                surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantages[i]
                
                # Policy + Value + Entropy Loss
                loss = -torch.min(surr1, surr2) + 0.5 * self.mse_loss(curr_value, returns[i]) - self.entropy_coef * entropy
                
                total_loss += loss
                
            # Step Optimizer
            self.optimizer.zero_grad()
            (total_loss / len(memory)).backward()
            self.optimizer.step()

        return total_loss.item() / len(memory)

def train_scientist_on_lab(
    num_episodes=50, 
    steps_per_episode=10, 
    num_agents=1
):
    """
    Core entry point for training the AI Scientist's Brain.
    """
    from marl_scientist.env.lab_env import LabEnvironment
    from marl_scientist.knowledge.real_store import RealKnowledgeStore
    
    # 1. Components
    brain = MetaBrain()
    trainer = MetaPPOTrainer(brain)
    encoder = BrainEncoder()
    kb = RealKnowledgeStore()
    kb.ingest_folder("knowledge/")
    
    lab = LabEnvironment(authorized_benchmarks=["CartPole-v1", "Acrobot-v1", "Pendulum-v1"])
    
    print(f"Starting Meta-Training: {num_episodes} episodes...")
    
    for ep in range(num_episodes):
        memory = []
        hidden_states = {f"Agent_{i}": None for i in range(num_agents)}
        
        # Reset lab/history for each training episode? 
        # Usually yes, or keep running history for 'curriculum'.
        
        for step in range(steps_per_episode):
            obs = lab.get_observation()
            
            step_actions = {}
            step_mem_entries = {}
            
            for i in range(num_agents):
                agent_id = f"Agent_{i}"
                
                # Get Knowledge context
                # Simplified: use random or constant query for training
                knowledge_vec = np.zeros(768) 
                
                inputs = encoder.encode_observation(obs, agent_id, knowledge_vec)
                
                # Select Action
                act_data = trainer.select_action(inputs, hidden_states[agent_id])
                hidden_states[agent_id] = act_data["new_hidden"]
                
                # Decode to Config
                config_dict = encoder.decode_action(act_data["algo_idx"], act_data["env_idx"], act_data["hp_vals"])
                config = ExperimentConfig(**config_dict)
                step_actions[agent_id] = config
                
                # Store entry (sans reward for now)
                step_mem_entries[agent_id] = {
                    "inputs": inputs,
                    "hidden": act_data["hidden"],
                    "algo_idx": act_data["algo_idx"],
                    "env_idx": act_data["env_idx"],
                    "hp_vals": act_data["hp_vals"],
                    "log_prob": act_data["log_prob"],
                    "value": act_data["value"]
                }
            
            # Lab Step
            results, rewards = lab.step(step_actions)
            
            # Post-step memory update
            for agent_id, mem in step_mem_entries.items():
                reward = rewards.get(agent_id, 0.0)
                mem["reward"] = reward
                mem["done"] = (step == steps_per_episode - 1)
                memory.append(mem)
                
        # Update Brain
        loss = trainer.update(memory)
        print(f"Episode {ep+1} complete. Avg Reward: {np.mean([m['reward'] for m in memory]):.2f}, Loss: {loss:.4f}")
        
        # Save checkpoint
        if (ep + 1) % 10 == 0:
            torch.save(brain.state_dict(), f"saves/scientist_brain_ep{ep+1}.pth")

if __name__ == "__main__":
    train_scientist_on_lab()
