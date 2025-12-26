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
        entropy_coef=0.05
    ):
        self.brain = brain
        self.optimizer = optim.Adam(self.brain.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.entropy_coef = entropy_coef
        
        self.mse_loss = nn.MSELoss()
        
        # [PHASE 2] LR Scheduler
        self.scheduler = optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=0.1, total_iters=100)
        
    def select_action(self, inputs: Dict[str, torch.Tensor], hidden_state: Optional[torch.Tensor]):
        """
        Samples actions from the brain and returns log_probs for training.
        """
        self.brain.eval()
        with torch.no_grad():
            algo_logits, env_logits, hp_means, value, new_hidden, goal_logits = self.brain(
                inputs["history"],
                inputs["trends"],
                inputs["knowledge"],
                inputs.get("latent"), # Pass latent context if available
                None, # memory_context
                None, # preference_vec (uses default)
                hidden_state
            )
            
            # 1. Goal selection
            dist_goal = Categorical(logits=goal_logits)
            goal_idx = dist_goal.sample()
            
            # 2. Algo selection
            dist_algo = Categorical(logits=algo_logits)
            algo_idx = dist_algo.sample()
            
            # 3. Env selection
            dist_env = Categorical(logits=env_logits)
            env_idx = dist_env.sample()
            
            # 4. HP selection
            hp_std = torch.exp(self.brain.hp_logstd)
            dist_hp = Normal(hp_means, hp_std)
            hp_vals = dist_hp.sample()
            
            # Pack log probs
            log_prob = dist_goal.log_prob(goal_idx) + \
                       dist_algo.log_prob(algo_idx) + \
                       dist_env.log_prob(env_idx) + \
                       dist_hp.log_prob(hp_vals).sum(dim=-1)
                       
        return {
            "goal_idx": goal_idx.item(),
            "algo_idx": algo_idx.item(),
            "env_idx": env_idx.item(),
            "hp_vals": hp_vals.squeeze(0).cpu().numpy(),
            "log_prob": log_prob.item(),
            "value": value.item(),
            "hidden": hidden_state,
            "new_hidden": new_hidden
        }

    def update(self, memory: List[Dict[str, Any]], gae_lambda: float = 0.95):
        """
        Performs PPO update on collected trajectories using GAE.
        """
        if not memory: return 0.0
        
        self.brain.train()
        
        # 1. Prepare Tensors
        rewards = [m['reward'] for m in memory]
        values = [m['value'] for m in memory]
        is_terminals = [m['done'] for m in memory]
        
        # [NEW] GAE Calculation
        returns = []
        advantages = []
        gae = 0
        last_value = 0 # In a real env, we might use the value of the next state if not terminal
        
        for i in reversed(range(len(rewards))):
            mask = 0 if is_terminals[i] else 1
            delta = rewards[i] + self.gamma * last_value * mask - values[i]
            gae = delta + self.gamma * gae_lambda * mask * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[i])
            last_value = values[i]
            
        returns = torch.tensor(returns, dtype=torch.float32).to(self.brain.hp_logstd.device)
        old_log_probs = torch.tensor([m['log_prob'] for m in memory], dtype=torch.float32).to(self.brain.hp_logstd.device)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.brain.hp_logstd.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        total_epoch_loss = 0
        for _ in range(self.k_epochs):
            total_loss = 0
            
            # Handle hidden states properly
            for i, m in enumerate(memory):
                inputs = m['inputs']
                h = m['hidden']
                
                # Move inputs to device
                d_inputs = {k: v.to(self.brain.hp_logstd.device) for k, v in inputs.items()}
                if h is not None: h = h.to(self.brain.hp_logstd.device)
                
                curr_algo_logits, curr_env_logits, curr_hp_means, curr_value, _, curr_goal_logits = self.brain(
                    d_inputs["history"],
                    d_inputs["trends"],
                    d_inputs["knowledge"],
                    d_inputs.get("latent"),
                    None, None,
                    h
                )
                
                # Distributions
                d_goal = Categorical(logits=curr_goal_logits)
                d_algo = Categorical(logits=curr_algo_logits)
                d_env = Categorical(logits=curr_env_logits)
                d_hp = Normal(curr_hp_means, torch.exp(self.brain.hp_logstd))
                
                # New log probs of the OLD actions
                lp_goal = d_goal.log_prob(torch.tensor([m['goal_idx']], device=self.brain.hp_logstd.device))
                lp_algo = d_algo.log_prob(torch.tensor([m['algo_idx']], device=self.brain.hp_logstd.device))
                lp_env = d_env.log_prob(torch.tensor([m['env_idx']], device=self.brain.hp_logstd.device))
                lp_hp = d_hp.log_prob(torch.tensor(m['hp_vals'], device=self.brain.hp_logstd.device)).sum(dim=-1)
                
                new_lp = lp_goal + lp_algo + lp_env + lp_hp
                entropy = d_goal.entropy() + d_algo.entropy() + d_env.entropy() + d_hp.entropy().sum()
                
                # PPO Ratio
                ratio = torch.exp(new_lp - old_log_probs[i])
                
                # Surrogate Loss
                surr1 = ratio * advantages[i]
                surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantages[i]
                
                # Policy + Value + Entropy Loss
                # Fix broadcasting: curr_value is [1, 1], returns[i] is scalar
                val_loss = 0.5 * self.mse_loss(curr_value, returns[i].view(1, 1))
                loss = -torch.min(surr1, surr2) + val_loss - self.entropy_coef * entropy
                
                total_loss += loss
                
            # [PHASE 2] Stability Metrics
            with torch.no_grad():
                avg_entropy = entropy.item() if 'entropy' in locals() else 0.0
            
            # Step Optimizer
            self.optimizer.zero_grad()
            (total_loss / len(memory)).backward()
            # [NEW] Gradient Clipping
            torch.nn.utils.clip_grad_norm_(self.brain.parameters(), max_norm=0.5)
            self.optimizer.step()
            self.scheduler.step()
            total_epoch_loss += total_loss.item()

        return (total_epoch_loss / (self.k_epochs * len(memory))), avg_entropy

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
