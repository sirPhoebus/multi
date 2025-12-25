from typing import Dict, Any, List, Optional
import random
import numpy as np
import dataclasses
from marl_scientist.core import Researcher, Observation, ExperimentConfig, ExperimentResult
from marl_scientist.agents.causal_graph import CausalDiscoveryEngine

class ResearcherAgent(Researcher):
    """
    An autonomous researcher agent.
    Uses Causal Discovery to learn algorithm dynamics.
    """
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        # "Scientific Beliefs" - current best known config
        self.current_best_config = ExperimentConfig(
            algorithm="PPO",
            hyperparameters={
                "learning_rate": 0.0003,
                "gamma": 0.99,
                "ent_coef": 0.0,
                "net_arch": [64, 64],
                "activation_fn": "Tanh"
            }
        )
        self.best_performance = -float('inf')
        self.best_performance = -float('inf')
        self.causal_model = CausalDiscoveryEngine()
        self.knowledge_store = None # Injected later
        self.knowledge_store = None # Injected later
        
    def set_knowledge_store(self, store):
        self.knowledge_store = store
        
    def propose_experiment(self, observation: Observation) -> ExperimentConfig:
        """
        Decide what to try next.
        """
        # 1. Literature Review: Check the shared knowledge base
        literature_config = self._review_literature()
        if literature_config:
            # If we found a great paper, maybe try to replicate/improve it
            if random.random() < 0.4: # 40% chance to follow literature
                 return self._mutate_config(literature_config, observation)

        # [NEW] Log Observation Features for debugging/verification
        if observation.performance_trends:
             pt = observation.performance_trends
             nl = observation.novelty_landscape
             # print(f"[Agent {self.agent_id}] Obs: Trend={pt.get('improvement_rate',0):.2f}, Stability={pt.get('stability',0):.2f}, Explored={nl.get('explored_ratio',0):.2f}")

        # 2. Social Learning: Check if anyone else (immediate peers in this batch) found something amazing
        best_peer_config = self._check_peer_results(observation)
        if best_peer_config:
             # Adopt peer's strategy as new baseline if it's much better
             # (In a real system, we'd verify it first)
             self.current_best_config = best_peer_config

        # 3. Causal Reasoning: Use internal model to improve current config
        # Epsilon-greedy: data gathering vs exploitation
        if random.random() < 0.2:
             # Random Exploration (Mutation or totally new random config)
             if random.random() < 0.5:
                 return self._random_config(observation) # Wide exploration
             else:
                 return self._mutate_config(self.current_best_config, observation) # Local exploration
        else:
            # "Reasoned" proposal
            new_params = self.causal_model.suggest_improvements(self.current_best_config.hyperparameters)
            return ExperimentConfig(
                algorithm=self.current_best_config.algorithm,
                hyperparameters=new_params,
                env_id=self.current_best_config.env_id # Keep same env
            )
            
    def update_knowledge(self, result: ExperimentResult):
        """
        Learn from the result.
        """
        # Update Causal Model
        # Update Causal Model
        self.causal_model.update_model(
            result.config.hyperparameters, 
            result.metrics, 
            result.final_mean_reward
        )
        
        # Update Best Known
        if result.final_mean_reward > self.best_performance:
            self.best_performance = result.final_mean_reward
            self.current_best_config = result.config
            
            # Publish if it's a significant finding (e.g. > 300)
            if self.best_performance > 300.0 and self.knowledge_store:
                paper = self.knowledge_store.synthesize_new_paper(result, self.agent_id)
                self.knowledge_store.add_paper(paper)

    def _review_literature(self) -> Optional[ExperimentConfig]:
        """
        Consults the agent's shard of the Knowledge Base.
        """
        if not self.knowledge_store: return None
        
        # 1. Contextual Query Generation
        # Analyze recent performance to formulate a query
        intent = random.choice(["general", "stability", "exploration", "efficiency"])
        
        base_algo = self.current_best_config.algorithm if self.current_best_config else "PPO"
        
        if intent == "stability":
            query = f"Stabilizing {base_algo} training, reducing variance"
        elif intent == "exploration":
            query = f"Exploration strategies and entropy bonus for {base_algo}"
        elif intent == "efficiency":
            query = f"Sample efficient RL with {base_algo}, faster convergence"
        else:
            query = f"Improving {base_algo} performance on control tasks"
            
        # Add random flavor keywords to query to force vector search diversity
        flavor = random.choice(["", "SOTA", "robust", "optimization", "parameters"])
        if flavor: query += f" {flavor}"
        
        # 2. Search (Local + Journal)
        # We increase k to get more diversity
        results = self.knowledge_store.search(query, k=5)
        
        if not results: return None
        
        # Probabilistic Selection (Diversity)
        # Instead of picking [0], we pick from top-3 or top-k with exponentially decreasing probability
        # Or simple: pick random from top 3
        selection_idx = min(len(results)-1, random.choices([0, 1, 2], weights=[0.6, 0.3, 0.1])[0])
        top_match = results[selection_idx]
        
        meta = top_match['metadata']
        text = top_match['text']
        
        print(f"Agent {self.agent_id} read paper ({intent}): '{meta['title']}'")
        
        # 3. Use LLM to Hypothesize a change (RAG)
        # We ask the KnowledgeStore to synthesize a config for us
        try:
            # We access the store directly. Note: The injected store is a "KnowledgeShard" which might not have the LLM method helper
            # So we might need to call it on self.knowledge_store.parent_store OR add it to shard.
            # Shard has .client and .parent_store
            
            # Let's check type. 
            # If it's a Shard, we use parent_store? Or just implement it in Shard?
            # It's cleaner if the Shard exposes it or we access parent.
            # Assuming set_knowledge_store passed a KnowledgeShard which wraps parent_store
            parent = getattr(self.knowledge_store, "parent_store", self.knowledge_store)
            
            # Synthesize
            llm_config_dict = parent.suggest_config_from_paper(text, paper_title=meta.get('title', 'Unknown'))
            
            if llm_config_dict:
                # Merge with our defaults to ensure validity (e.g. if LLM missed net_arch)
                # But let's trust the Algo + HP from LLM
                
                # Careful: We need to respect the Expanded Space format
                # Ensure HPs are simple types
                
                return ExperimentConfig(
                    algorithm=llm_config_dict["algorithm"],
                    hyperparameters=llm_config_dict["hyperparameters"],
                    env_id=llm_config_dict.get("env_id", self.current_best_config.env_id if self.current_best_config else "CartPole-v1")
                )
        except Exception as e:
            print(f"Agent {self.agent_id} failed to synthesize config from paper: {e}")

        return None
            
    def _check_peer_results(self, observation: Observation) -> Any:
        # Simple scan of history for high performing results
        if not observation.experiment_history: return None
        
        # Sort by reward
        sorted_hist = sorted(observation.experiment_history, key=lambda x: x.final_mean_reward, reverse=True)
        top_result = sorted_hist[0]
        
        # Adaptive Threshold:
        # If we have a very low score, we should be eager to copy (low threshold).
        # If we have a high score, we should be picky (high threshold).
        # Logic: Peer must be at least X% better, or absolute Y better if we are just starting.
        
        current_score = self.best_performance if self.best_performance > -float('inf') else 0.0
        
        # Calculate dynamic margin: 10% improvement or 20 points, whichever is larger
        # This prevents switching for trivial noise (e.g. 495 vs 498) but allows jumps early on.
        margin = max(20.0, current_score * 0.10)
        
        if top_result.final_mean_reward > current_score + margin:
             # print(f"Agent {self.agent_id} adopting peer config (Peer: {top_result.final_mean_reward:.1f} vs Self: {current_score:.1f})")
             return top_result.config
        return None

    def _random_config(self, observation: Optional[Observation] = None) -> ExperimentConfig:
        """Generate a completely random valid configuration."""
        # NEW in Step 9: Select Environment first, then algo
        available_envs = list(observation.env_metadata.keys()) if observation and observation.env_metadata else ["CartPole-v1"]
        env_id = random.choice(available_envs)
        
        # Check env constraints
        is_continuous = False
        if observation and env_id in observation.env_metadata:
             is_continuous = observation.env_metadata[env_id].get("is_continuous", False)
        
        if is_continuous:
             algo = random.choice(["SAC", "PPO", "A2C"])
        else:
             algo = random.choice(["PPO", "A2C", "DQN"])
        
        hp = {}
        
        # Common Params
        hp["learning_rate"] = random.choice([1e-3, 5e-4, 3e-4, 1e-4, 5e-5, 1e-5])
        hp["gamma"] = random.choice([0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
        
        # Advanced Common Params
        hp["gae_lambda"] = random.choice([0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
        hp["vf_coef"] = random.choice([0.1, 0.5, 1.0])
        hp["max_grad_norm"] = random.choice([0.3, 0.5, 1.0, 5.0])
        
        # Architecture
        # Support for simple lists and dictionary (asymetric) architectures for SB3
        arch_type = random.choice(["simple", "deep", "wide", "funnel", "asymmetric"])
        
        if arch_type == "simple":
            hp["net_arch"] = [64, 64]
        elif arch_type == "deep":
            hp["net_arch"] = [64, 64, 64, 64]
        elif arch_type == "wide":
            hp["net_arch"] = [256, 256]
        elif arch_type == "funnel":
            hp["net_arch"] = [128, 64, 32]
        elif arch_type == "asymmetric":
            # Separate Actor (pi) and Critic (vf) networks
            # SB3 expects: dict(pi=[...], vf=[...])
            # We encode this as a special struct or just a dict that our runner handles
            hp["net_arch"] = {"pi": [64, 64], "vf": [128, 128]}

        hp["activation_fn"] = random.choice(["ReLU", "Tanh", "ELU", "LeakyReLU"])
        
        # Algo Specifics
        if algo == "PPO":
            hp["n_steps"] = random.choice([128, 256, 512, 1024, 2048])
            hp["batch_size"] = random.choice([32, 64, 128])
            hp["n_epochs"] = random.choice([3, 5, 10, 20])
            hp["clip_range"] = random.choice([0.1, 0.2, 0.3])
            hp["ent_coef"] = random.uniform(0.0, 0.05)
            hp["target_kl"] = random.choice([None, 0.01, 0.02, 0.05])
            
        elif algo == "A2C":
            hp["n_steps"] = random.choice([5, 10, 20, 50])
            hp["ent_coef"] = random.uniform(0.0, 0.05)
            
        elif algo == "DQN":
            hp["buffer_size"] = random.choice([10000, 50000, 100000])
            hp["learning_starts"] = random.choice([100, 1000, 5000])
            hp["batch_size"] = random.choice([32, 64, 128])
            hp["target_update_interval"] = random.choice([500, 1000, 5000])
            hp["exploration_fraction"] = random.uniform(0.1, 0.5)
            hp["exploration_final_eps"] = random.choice([0.01, 0.02, 0.05])
            
        elif algo == "SAC":
            hp["buffer_size"] = random.choice([10000, 50000, 100000])
            hp["batch_size"] = random.choice([64, 128, 256])
            hp["learning_starts"] = random.choice([100, 1000])
            hp["tau"] = random.choice([0.005, 0.01, 0.02, 0.05])
            hp["train_freq"] = random.choice([1, 4, 8])
            hp["gradient_steps"] = random.choice([1, 2, 4]) # Steps per update
            hp["ent_coef"] = "auto" 
            
        # Select Environment moved to top of _random_config
        return ExperimentConfig(algorithm=algo, hyperparameters=hp, env_id=env_id)

    def _mutate_config(self, base_config: ExperimentConfig, observation: Optional[Observation] = None) -> ExperimentConfig:
        """Apply random mutations to hyperparameters or architecture."""
        if not base_config:
            return self._random_config(observation)
            
        # 20% chance to jump to a completely new random config (Increased Exploration)
        if random.random() < 0.2:
            return self._random_config(observation)
            
        hp = base_config.hyperparameters.copy()
        
        # Mutation Strategy
        keys = list(hp.keys())
        # Weighted choice to prefer mutating numeric params over architecture
        mutation_target = random.choice(keys + ["ARCH_SWAP", "PARAM_JITTER"]) 
        
        if mutation_target == "ARCH_SWAP":
            if random.random() < 0.5:
                # Swapping topology
                arch_type = random.choice(["simple", "deep", "wide", "funnel", "asymmetric"])
                if arch_type == "simple": hp["net_arch"] = [64, 64]
                elif arch_type == "deep": hp["net_arch"] = [64, 64, 64, 64]
                elif arch_type == "wide": hp["net_arch"] = [256, 256]
                elif arch_type == "funnel": hp["net_arch"] = [128, 64, 32]
                elif arch_type == "asymmetric": hp["net_arch"] = {"pi": [64, 64], "vf": [128, 128]}
            else:
                 # Change Activation
                 hp["activation_fn"] = random.choice(["ReLU", "Tanh", "ELU", "LeakyReLU"])

        elif mutation_target == "PARAM_JITTER":
             # Jitter a random numeric parameter
             numeric_keys = [k for k in keys if isinstance(hp[k], (int, float)) and not isinstance(hp[k], bool)]
             if numeric_keys:
                 k = random.choice(numeric_keys)
                 val = hp[k]
                 if isinstance(val, float):
                     hp[k] = val * random.uniform(0.8, 1.2)
                 elif isinstance(val, int):
                     # For ints, we need to be careful (e.g. batch size)
                     # Maybe just flip to a neighbour value in a predefined list?
                     # For now, small additive noise rounded
                     change = random.choice([-1, 1]) * max(1, int(val * 0.1))
                     hp[k] = max(1, val + change)

        elif mutation_target in hp:
             # Point mutation of specific key
             val = hp[mutation_target]
             if isinstance(val, (int, float)):
                 hp[mutation_target] = val * random.uniform(0.8, 1.2)
             elif isinstance(val, str):
                 # Categorical flip?
                 pass 
                 
        return ExperimentConfig(
            algorithm=base_config.algorithm,
            hyperparameters=hp,
            env_id=base_config.env_id # CRITICAL: Propagate env_id
        )

    def save(self, filepath: str):
        """Persist agent state."""
        import pickle
        import os
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        data = {
            "agent_id": self.agent_id,
            "current_best_config": self.current_best_config,
            "best_performance": self.best_performance
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
            
        # Also save causal model
        if self.causal_model:
            self.causal_model.save_state(filepath + ".causal")
            
        print(f"[Agent {self.agent_id}] State saved to {filepath}")

    def load(self, filepath: str):
        """Load agent state."""
        import pickle
        import os
        
        if not os.path.exists(filepath):
            return
            
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                
            self.agent_id = data.get("agent_id", self.agent_id)
            self.current_best_config = data.get("current_best_config", self.current_best_config)
            self.best_performance = data.get("best_performance", self.best_performance)
            
            # Load causal model
            if self.causal_model:
                self.causal_model.load_state(filepath + ".causal")
                
            print(f"[Agent {self.agent_id}] State loaded. Best Reward: {self.best_performance:.1f}")
        except Exception as e:
            print(f"[Agent {self.agent_id}] Failed to load state: {e}")
