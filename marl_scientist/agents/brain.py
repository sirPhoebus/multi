import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

class MetaBrain(nn.Module):
    """
    The 'Neural Brain' for a Researcher Agent.
    Implements a sequence-aware model that proposes experiments.
    """
    def __init__(
        self, 
        history_dim: int = 68,  # Increased [64 -> 68] for temporal features
        trend_dim: int = 8, 
        knowledge_dim: int = 768, # Matches LLM embeddings
        hidden_dim: int = 256,
        num_algos: int = 4,
        num_envs: int = 5,
        num_continuous_hps: int = 6
    ):
        super().__init__()
        
        # 1. History Encoder (LSTM)
        self.history_lstm = nn.LSTM(input_size=history_dim, hidden_size=hidden_dim, batch_first=True)
        
        # 2. Trends Encoder
        self.trends_mlp = nn.Sequential(
            nn.Linear(trend_dim, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_dim)
        )
        
        # 3. Knowledge Encoder
        self.knowledge_mlp = nn.Sequential(
            nn.Linear(knowledge_dim, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim)
        )
        
        # 4. Fusion & Decision Core
        # Trends (hidden_dim) + Knowledge (hidden_dim) + RNN (hidden_dim) + [NEW] MemoryContext (6) + [NEW] Preference (3)
        self.obs_dim = hidden_dim * 3 + 6 + 3 
        
        self.fusion = nn.Linear(self.obs_dim, hidden_dim)
        self.decision_rnn = nn.GRUCell(hidden_dim, hidden_dim) 

        # [NEW] Hierarchical Goal Head (Manager)
        # 3 Goals: 0=EXPLORE, 1=EXPLOIT, 2=REFINE
        self.goal_head = nn.Linear(hidden_dim, 3) 
        
        # Worker Fusion (Conditioned on Goal)
        self.worker_fusion = nn.Linear(hidden_dim + 3, hidden_dim)
        
        # 5. Policy Heads (Now Worker Heads)
        self.algo_head = nn.Linear(hidden_dim, num_algos)
        self.env_head = nn.Linear(hidden_dim, num_envs)
        
        # For HPs, we output mean for each
        self.hp_mean_head = nn.Linear(hidden_dim, num_continuous_hps)
        # Shared logstd for exploration
        self.hp_logstd = nn.Parameter(torch.zeros(num_continuous_hps))
        
        # 6. Value Head
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(
        self, 
        history_seq: torch.Tensor,     # (batch, seq_len, history_dim)
        trends: torch.Tensor,          # (batch, trend_dim)
        knowledge: torch.Tensor,       # (batch, knowledge_dim)
        memory_context: torch.Tensor = None, # (batch, 6)
        preference_vec: torch.Tensor = None, # [NEW] (batch, 3) -> [Perf, Efficiency, Stability]
        hidden_state: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        batch_size = trends.size(0)
        
        # Encoder passes
        if history_seq.shape[1] > 0:
            _, (h_hist, _) = self.history_lstm(history_seq)
            h_hist = h_hist[-1] 
        else:
            h_hist = torch.zeros(batch_size, self.history_lstm.hidden_size, device=history_seq.device)
            
        h_trends = self.trends_mlp(trends)
        h_know = self.knowledge_mlp(knowledge)
        
        # Handle Missing Contexts
        if memory_context is None:
            memory_context = torch.zeros(batch_size, 6, device=trends.device)
        if preference_vec is None:
            preference_vec = torch.tensor([[0.5, 0.25, 0.25]], device=trends.device).repeat(batch_size, 1)
        
        # Fusion
        combined = torch.cat([h_hist, h_trends, h_know, memory_context, preference_vec], dim=-1)
        latent = F.relu(self.fusion(combined))
        
        # Session state update
        if hidden_state is None:
            hidden_state = torch.zeros_like(latent)
        
        new_hidden = self.decision_rnn(latent, hidden_state)

        # [NEW] Hierarchical Strategic Step
        goal_logits = self.goal_head(new_hidden)
        goal_selected = torch.softmax(goal_logits, dim=-1) # soft-conditioning for training
        
        # Worker Decisions conditioned on Goal
        worker_input = torch.cat([new_hidden, goal_selected], dim=-1)
        worker_latent = F.relu(self.worker_fusion(worker_input))
        
        # Heads (Worker)
        algo_logits = self.algo_head(worker_latent)
        env_logits = self.env_head(worker_latent)
        hp_means = torch.tanh(self.hp_mean_head(worker_latent)) 
        
        value = self.value_head(worker_latent)
        
        return algo_logits, env_logits, hp_means, value, new_hidden, goal_logits

class BrainEncoder:
    """Helper to convert core objects to tensors for the Brain."""
    
    ALGOS = ["PPO", "A2C", "DQN", "SAC"]
    ENVS = ["CartPole-v1", "LunarLander-v3", "Pendulum-v1", "Acrobot-v1", "MountainCarContinuous-v0"]
    
    def __init__(self, history_len: int = 10):
        self.history_len = history_len

    def encode_observation(
        self, 
        observation: Any, 
        agent_id: str, 
        knowledge_vec: Optional[np.ndarray] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Converts the rich Observation object into fixed-size tensors.
        """
        # 1. Trends (trend_dim=8)
        pt = observation.performance_trends
        nl = observation.novelty_landscape
        
        trends_vec = np.array([
            pt.get("mean_reward_all", 0) / 500.0,
            pt.get("improvement_rate", 0) / 10.0,
            pt.get("stability", 0),
            nl.get("explored_ratio", 0),
            nl.get("archive_size", 0) / 100.0,
            0.0, 0.0, 0.0 # Padding
        ], dtype=np.float32)

        # 2. History (history_dim=64)
        # We take the last N experiments where this agent was involved, or all recent ones.
        # For now, let's take ALL recent experiments to see what's working globaly.
        hist_seq = []
        recent_hist = observation.experiment_history[-self.history_len:]
        
        for res in recent_hist:
            # Algo One-Hot (4)
            algo_idx = self.ALGOS.index(res.config.algorithm) if res.config.algorithm in self.ALGOS else 0
            algo_one_hot = np.zeros(4)
            algo_one_hot[algo_idx] = 1.0
            
            # Env One-Hot (4)
            env_idx = self.ENVS.index(res.config.env_id) if res.config.env_id in self.ENVS else 0
            env_one_hot = np.zeros(4)
            env_one_hot[env_idx] = 1.0
            
            # Outcome (1)
            reward = np.array([res.final_mean_reward / 500.0])
            
            # HPs (Let's encode the main ones) (8)
            hp = res.config.hyperparameters
            lr = np.log10(hp.get("learning_rate", 3e-4)) / 5.0
            gamma = hp.get("gamma", 0.99)
            ent = hp.get("ent_coef", 0.0) * 10.0
            
            hp_vec = np.array([lr, gamma, ent, 0, 0, 0, 0, 0])
            
            # Combine (4+4+1+8 = 17)
            # [NEW] Temporal Features (2 added)
            # Duration: log scale, assume ~60s baseline
            dur = getattr(res, "duration_seconds", 10.0)
            dur_norm = np.log1p(dur) / np.log1p(60.0)
            
            # Steps: normalize by 500k
            steps = getattr(res, "total_env_steps", 10_000)
            steps_norm = steps / 500_000.0
            
            # 17 + 2 = 19 features total so far
            # Pad to dynamic history_dim (usually 68 now)
            # Explicitly define vector size from class attr if possible, or just pad hard
            target_dim = 68 # Synced with MetaBrain init
            
            combined = np.concatenate([algo_one_hot, env_one_hot, reward, hp_vec, [dur_norm, steps_norm]])
            padded = np.zeros(target_dim)
            padded[:len(combined)] = combined
            hist_seq.append(padded)
            
        if not hist_seq:
            # Handle empty history
            hist_seq = [np.zeros(68)]
            
        # 3. Knowledge (768)
        if knowledge_vec is None:
            knowledge_vec = np.zeros(768)
            
        return {
            "history": torch.tensor(np.array(hist_seq), dtype=torch.float32).unsqueeze(0),
            "trends": torch.tensor(trends_vec, dtype=torch.float32).unsqueeze(0),
            "knowledge": torch.tensor(knowledge_vec, dtype=torch.float32).unsqueeze(0)
        }

    def decode_action(self, algo_idx: int, env_idx: int, hp_vals: np.ndarray) -> Dict[str, Any]:
        """Maps NN outputs back to ExperimentConfig fields."""
        algo = self.ALGOS[algo_idx]
        env = self.ENVS[env_idx]
        
        # Map [-1, 1] to specific ranges
        def scale(val, min_val, max_val, log=False):
            if log:
                lmin, lmax = np.log10(min_val), np.log10(max_val)
                res = 10**((val + 1) / 2 * (lmax - lmin) + lmin)
                return res
            else:
                return (val + 1) / 2 * (max_val - min_val) + min_val

        hps = {
            "learning_rate": float(scale(hp_vals[0], 1e-5, 1e-2, log=True)),
            "gamma": float(scale(hp_vals[1], 0.8, 0.9999)),
            "ent_coef": float(scale(hp_vals[2], 0.0, 0.1)),
            "gae_lambda": float(scale(hp_vals[3], 0.8, 1.0)),
            "total_timesteps": 30000 # Increased for realistic exams
        }
        
        # Add algo-specific defaults/scaling
        if algo == "PPO":
            hps["n_steps"] = int(scale(hp_vals[4], 128, 2048))
            hps["batch_size"] = int(scale(hp_vals[5], 32, 512))
        elif algo == "DQN":
            hps["batch_size"] = int(scale(hp_vals[4], 32, 256))
            hps["learning_starts"] = int(scale(hp_vals[5], 100, 5000))
        elif algo == "SAC":
             hps["gradient_steps"] = 1 # Force 1 update per step for speed
             hps["tau"] = float(scale(hp_vals[5], 0.005, 0.05))
             hps["batch_size"] = int(scale(hp_vals[4], 64, 256)) # Reuse hp[4] since grad_steps removed
             hps["total_timesteps"] = 30000 # Realistic duration
        
        return {"algorithm": algo, "env_id": env, "hyperparameters": hps}
