from typing import Dict, Any, List, Optional, Tuple, Union
import random
import numpy as np
import dataclasses
from marl_scientist.core import Researcher, Observation, ExperimentConfig, ExperimentResult
from marl_scientist.agents.causal_graph import CausalDiscoveryEngine
from marl_scientist.llm.client import LLMClient
import re

from marl_scientist.utils.logger import setup_logger

class ResearcherAgent(Researcher):
    """
    An autonomous researcher agent.
    Uses Causal Discovery to learn algorithm dynamics.
    """
    
    def __init__(self, agent_id: str, knowledge_store: Optional[Any] = None, trajectory_memory: Optional[Any] = None):
        self.agent_id = agent_id
        self.knowledge_store = knowledge_store
        self.trajectory_memory = trajectory_memory
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
        self.causal_model = CausalDiscoveryEngine()
        self.log = setup_logger()
        self.validator = None # [PHASE 2]
        
        # [NEW] LLM for Skills
        self.llm = LLMClient()
        self.llm_available = False # Will check on first use or init
        # Lazy check to avoid blocking init
        # if self.llm.check_connection(): self.llm_available = True
        
        # [NEW] Meta-Learning Competencies
        # domain -> score (0.0 to 1.0)
        self.competence = {"rl": 0.1, "coding": 0.1} 
        self.competence_alpha = 0.1 # Learning rate for competence
        
    def set_knowledge_store(self, store):
        self.knowledge_store = store
    
    def set_trajectory_memory(self, memory):
        self.trajectory_memory = memory

    def set_validator(self, validator):
        self.validator = validator
        
    async def propose_experiment(self, observation: Observation) -> Tuple[Union[ExperimentConfig, List[ExperimentConfig]], Dict[str, Any]]:
        """
        Decide what to try next using a hybrid approach.
        """
        # [NEW] Trigger slow reasoning on high uncertainty or stagnation
        # User defined triggers
        high_uncertainty = observation.novelty_landscape.get("unexplored_ratio", 0.0) > 0.7
        stagnant = self.competence_low()
        
        if high_uncertainty or stagnant:
            self.log.info(f"[Agent {self.agent_id}] Slow Reasoning Triggered: Uncertainty={high_uncertainty}, Stagnation={stagnant}")
            # 1. Literature Review & Context Gathering
            context = await self._gather_context(observation)
            
            # 2. Generate Hypothesis
            hypothesis = await self._generate_hypothesis(observation, context)
            self.log.info(f"[Agent {self.agent_id}] Hypothesis: {hypothesis}")
            
            # 3. Translate Hypothesis to Experiment(s)
            proposal = await self._translate_hypothesis_to_experiment(hypothesis, observation)
            if proposal:
                 return proposal, {}

        # Fallback to standard mutation/random
        return self._mutate_config(self.current_best_config, observation), {}

    def competence_low(self) -> bool:
        """Check if overall competence is low across domains."""
        avg_comp = sum(self.competence.values()) / len(self.competence) if self.competence else 0.0
        return avg_comp < 0.2

    async def _gather_context(self, observation: Observation) -> str:
        """[PILLAR 1] Gathers literature and peer results into a context string."""
        context = "Recent peer results from Shared Journal:\n"
        if self.knowledge_store:
            # Query journal for insights related to current best algorithm
            query = f"Optimizing {self.current_best_config.algorithm} performance"
            journal_hits = await self.knowledge_store.search(query, k=3)
            if journal_hits:
                for h in journal_hits:
                    context += f"- {h['metadata'].get('title', 'Insight')}: {h['text']}\n"
        
        lit_hits = await self._review_literature_hits()
        if lit_hits:
            context += "\nRelevant literature from Knowledge Store:\n"
            for h in lit_hits:
                context += f"- {h['metadata']['title']}: {h['text'][:200]}...\n"
        
        return context

    def _check_peer_results(self, observation: Observation) -> Optional[ExperimentConfig]:
        """Checks the immediate observation for any peer who outperformed us in this session."""
        if not observation or not observation.metrics:
            return None
        
        best_reward = self.best_performance
        best_cfg = None
        
        for aid, metrics in observation.metrics.items():
            if aid == self.agent_id: continue
            reward = metrics.get("mean_reward", -float('inf'))
            if reward > best_reward:
                best_reward = reward
                # We assume the config is available in observation or we just know 
                # (In this simulation, we can peek if we want to simulate social learning)
                # But it's cleaner to use the KnowledgeStore as the source of truth.
        return best_cfg

    async def _generate_hypothesis(self, observation: Observation, context: str) -> str:
        """Uses LLM to generate a natural language hypothesis."""
        prompt = f"""
You are a Lead RL Researcher. Based on the following context and current state, propose a specific, testable hypothesis.

Current Best Config: {self.current_best_config}
Performance: {self.best_performance:.1f}
Context:
{context}

Format your response as: "Hypothesis: [Your single-sentence hypothesis]"
Encourage innovation: If performance is plateauing, suggest structural changes (e.g., "Use a Transformer-based policy", "Add a larger recurrent memory", "Use an ensemble of models").
"""
        try:
            response = await self.llm.async_chat_completion([{"role": "user", "content": prompt}])
            return response.strip()
        except:
            return "Hypothesis: Increasing exploration will improve performance in sparse-reward environments."

    async def _translate_hypothesis_to_experiment(self, hypothesis: str, observation: Observation) -> ExperimentConfig:
        """Translates an NL hypothesis into a concrete ExperimentConfig."""
        # For simplicity, we use the existing RAG-style synthesis logic or similar LLM call
        prompt = f"""
Translate the following hypothesis into a concrete experiment configuration.

Hypothesis: {hypothesis}
Target Domain: {observation.env_metadata.get(observation.env_id, {}).get("domain", "rl")}

Output a JSON object:
{{
  "algorithm": "PPO" | "A2C" | "DQN" | "SAC" | "TD3" | "DCC",
  "hyperparameters": {{ 
      ... ,
      "policy_code": "[Optional] For RL: Python code for a custom Stable Baselines3 Policy class",
      "model_code": "[Optional] For Vision: Python code for a custom PyTorch nn.Module class named 'CustomModel'"
  }},
  "env_id": "CartPole-v1" | "Acrobot-v1" | "Pendulum-v1" | "LunarLander-v3" | "MountainCarContinuous-v0" | "CIFAR100-Task-0" | ... 
}}

Rules for code generation:
1. RL: Must inherit from `BasePolicy` or `ActorCriticPolicy`.
2. Vision: Must inherit from `nn.Module` and be named `CustomModel`.
3. Provide ONLY the code string inside the JSON value.
"""
        import json
        try:
            response = await self.llm.async_chat_completion([{"role": "user", "content": prompt}])
            # Basic cleanup (similar to KnowledgeStore logic)
            clean_json = re.search(r"\{.*\}", response, re.DOTALL).group(0)
            data = json.loads(clean_json)
            return ExperimentConfig(
                algorithm=data["algorithm"],
                hyperparameters=data["hyperparameters"],
                env_id=data.get("env_id", "CartPole-v1")
            )
        except:
             return None

    async def propose_system_change(self, observation: Observation) -> Optional[Dict]:
        """[SELf-MODIFICATION] Propose a change to the simulation parameters using LLM."""
        if self.best_performance < 400: # Only elites propose changes
            return None
            
        prompt = f"""
            You are an Elite RL Lead Researcher. Your current best performance is {self.best_performance:.1f}.
            The simulation is running with:
            - update_interval (batch size)
            - meta_reward_weights (Perf, Efficiency, Stability)
            - population_cap
            
            Current Observation: {observation.performance_trends}
            
            Propose a single system-level modification to improve the collective swarm intelligence.
            Output JSON only:
            {{
                "target": "name_of_parameter",
                "value": new_value,
                "reason": "short_explanation"
            }}
        """
        try:
            response = await self.llm.async_chat_completion([{"role": "user", "content": prompt}])
            clean_json = re.search(r"\{.*\}", response, re.DOTALL).group(0)
            return json.loads(clean_json)
        except:
            return None
            
    async def update_knowledge(self, result: ExperimentResult):
        """
        Learn from the result.
        """
        # Update Causal Model
        self.causal_model.update_model(
            result.config.hyperparameters, 
            result.metrics, 
            result.final_mean_reward
        )
        
        # [NEW] Update Competence
        domain = result.config.domain
        score = result.performance_score # Normalized 0-1
        
        current = self.competence.get(domain, 0.1)
        # Exponential moving average
        new_competence = current + self.competence_alpha * (score - current)
        self.competence[domain] = new_competence
        
        # print(f"[Agent {self.agent_id}] Updated {domain} competence to {new_competence:.2f} (Rollout: {score:.2f})")
        
        # [PHASE 1] Trajectory Compression
        if self.trajectory_memory:
            await self.trajectory_memory.add(result)
        
        # [NEW] Deep Reflection (Research Papers)
        if not hasattr(self, 'recent_results'):
            self.recent_results = []
        self.recent_results.append(result)
        # Keep recent 20 for reflection
        if len(self.recent_results) > 20: self.recent_results.pop(0)

        # Update Best Known
        if result.final_mean_reward > self.best_performance:
            self.best_performance = result.final_mean_reward
            self.current_best_config = result.config
            
            # Publish if it's a significant finding
            if self.knowledge_store:
                paper = self.knowledge_store.synthesize_new_paper(result, self.agent_id)
                await self.knowledge_store.add_paper(paper)
                
                # [NEW] Add a concise insight
                insight = f"Agent {self.agent_id} optimized {result.config.algorithm} on {result.config.env_id} achieving {result.final_mean_reward:.1f}. Key params: {result.config.hyperparameters}"
                await self.knowledge_store.add_insight(insight, self.agent_id)

    async def perform_deep_reflection(self) -> Optional[str]:
        """Synthesizes recent trajectories into a structured Research Paper."""
        if not self.recent_results or not self.knowledge_store:
            return None
            
        self.log.info(f"[{self.agent_id}] Starting Deep Reflection Phase...")
        
        # Sample recent hits for summary
        summaries = [f"Env: {r.config.env_id}, Reward: {r.final_mean_reward:.1f}" for r in self.recent_results[-5:]]
        
        prompt = f"""
            You are a Senior AI Scientist. Reflect on your last series of experiments:
            {summaries}
            
            Write a formal, structured Research Paper in Markdown summarizing:
            - Goal: The objective of this research batch.
            - Observations: Patterns in hyperparameter sensitivity.
            - Meta-Insight: A high-level conclusion for the shared knowledge base.
            - Next Steps: Recommendations for future experiments.
            
            Keep it concise but impactful (300-500 words).
        """
        try:
            paper_content = await self.llm.async_chat_completion([{"role": "user", "content": prompt}])
            
            # Index as a paper in the knowledge store
            paper_obj = {
                "text": paper_content,
                "metadata": {
                    "title": f"Synthesis of Research Batch: Agent {self.agent_id}",
                    "author": self.agent_id,
                    "type": "reflection_paper"
                }
            }
            await self.knowledge_store.add_paper(paper_obj)
            self.recent_results = [] # Reset after reflection
            return paper_content
        except Exception as e:
            self.log.error(f"Reflection failed: {e}")
            return None

    async def _review_literature_hits(self) -> List[Dict]:
        """Internal helper for context gathering."""
        if not self.knowledge_store: return []
        query = f"Advanced strategies for {self.current_best_config.algorithm}"
        return await self.knowledge_store.search(query, k=3)
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
        
        # 2. Search (Local + Journal + [PHASE 1] Negative Knowledge)
        # We increase k to get more diversity
        results = self.knowledge_store.search(query, k=5)
        
        negative_context = ""
        if self.trajectory_memory:
             # Look for similar failures to avoid
             failures = self.trajectory_memory.get_failures(self.current_best_config, k=2)
             if failures:
                 negative_context = "\nAvoid these past failure modes:\n" + "\n".join([f"- {f}" for f in failures])
        
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
            llm_config_dict = parent.suggest_config_from_paper(
                text, 
                paper_title=meta.get('title', 'Unknown'),
                negative_knowledge=negative_context
            )
            
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
        
        if observation and env_id in observation.env_metadata:
             meta = observation.env_metadata[env_id]
             domain = meta.get("domain", "rl")
             is_continuous = meta.get("is_continuous", False)
        
        if domain == "coding":
            return self._generate_code_solution(env_id, observation)
        
        if domain == "vision":
             # Generate DCC Hyperparameters
             hp = {
                 "fast_lr": random.choice([1e-3, 5e-4, 1e-4]),
                 "slow_lr": random.choice([1e-3, 5e-4, 1e-4]),
                 "fast_weight": random.choice([0.5, 0.7, 0.9]),
                 "batch_size": random.choice([64, 128]),
                 "epochs": random.choice([5, 10]), # Short for lab
                 "input_dim": 3072,
                 "hidden_dim": random.choice([256, 512, 1024])
             }
             hp["slow_weight"] = 1.0 - hp["fast_weight"]
             
             # 50% chance to design a custom architecture instead of tuning DCC
             if self.competence.get("vision", 0) > 0.3 and random.random() < 0.5:
                 return self._generate_architecture_proposal(env_id, observation)
                 
             return ExperimentConfig(algorithm="DCC", hyperparameters=hp, env_id=env_id, domain="vision")
             
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
            
        if base_config.domain == "coding":
            # For coding, 'mutation' currently just means retry with valid syntax (random)
            # Future: Perturb code string
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

    def _select_target_domain(self) -> str:
        """Selects a domain to work on based on competence and curiosity."""
        # Simple Epsilon-Greedy for now
        # 20% explore random domain
        # 80% exploit high competence (or inverse if we want to improve weak skills?)
        # Let's say we want to IMPROVE weak skills? Or specialize?
        # The prompt says 'discover diverse skills'. So we should balance.
        # Let's use proportional sampling (Softmax-ish) but inverted?
        # Actually, let's just pick the one with highest potential or random.
        
        domains = list(self.competence.keys())
        if not domains: return "rl"
        
        if random.random() < 0.3:
             return random.choice(domains)
        else:
             # Pick best domain (Specialization)
             # return max(self.competence, key=self.competence.get)
             
             # Pick domain with moderate competence (Curriculum - not too easy, not too hard?)
             # For now, just random weighted by competence?
             return random.choices(domains, weights=[self.competence[d] + 0.1 for d in domains])[0]

    def _generate_code_solution(self, task_id: str, observation: Observation) -> ExperimentConfig:
        """Uses LLM to generate a Python solution for a coding task."""
        
        # 1. Get task description if available
        desc = "Write a python function named 'solution'."
        if observation and observation.env_metadata and task_id in observation.env_metadata:
             desc = observation.env_metadata[task_id].get("description", desc)
        
        prompt = f"""You are an expert Python programmer.
Task: {desc}

You must write a valid Python function named `solution`.
Return ONLY the python code code block. Do not explain.
"""
        
        # Fallback if no LLM
        # if not self.llm_available: # We could check this, but let's try calling it
        #    pass 
            
        try:
             # We assume self.llm is available or will handle connection errors gracefully
             response = self.llm.chat_completion([
                 {"role": "system", "content": "You are a concise python coding assistant."},
                 {"role": "user", "content": prompt}
             ], temperature=0.7)
             
             if not response:
                 raise Exception("Empty LLM response")
                 
             # Extract code from markdown blocks if present
             code = response
             if "```python" in response:
                 code = response.split("```python")[1].split("```")[0].strip()
             elif "```" in response:
                 code = response.split("```")[1].split("```")[0].strip()
                 
             # Safety: Ensure 'def solution' is in there
             if "def solution" not in code:
                  code = f"def solution(x):\n    # LLM failed to define solution\n    return x\n\n# Raw Output:\n# {code}"

        except Exception as e:
             # print(f"[Agent {self.agent_id}] LLM Code Gen Failed: {e}")
             # Fallback
             code = "def solution(x):\n    return x # Empty Fallback"
        
        return ExperimentConfig(
            algorithm="PythonScript", 
            hyperparameters={"code": code},
            env_id=task_id,
            domain="coding"
        )

    def _generate_architecture_proposal(self, task_id: str, observation: Observation) -> ExperimentConfig:
        """
        Designing a Neural Architecture for Vision.
        Recursive Self-Improvement Step 1: Modifying the compute structure.
        """
        system_prompt = (
            "You are an AI Architect designing a PyTorch Vision Model.\n"
            "Task: Create a neural network class named 'CustomModel' inheriting from nn.Module.\n"
            "Input: Flattened CIFAR-100 image (size 3072).\n"
            "Output: Class logits (size 5).\n"
            "Constraints:\n"
            "- Must accept an optional `hp` dict in __init__.\n"
            "- Must use standard PyTorch.\n"
            "- Be creative: use skip connections, dense blocks, or attention if useful.\n"
            "- Return ONLY the Python code block."
        )
        
        user_prompt = f"Design a model for task {task_id}. Make it better than a simple MLP."
        
        try:
            response = self.llm.chat_completion([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ], temperature=0.7)
        except Exception as e:
            print(f"LLM Architecture Gen failed: {e}")
            return ExperimentConfig(algorithm="DCC", hyperparameters={}, env_id=task_id, domain="vision")

        # Parsing code
        code_block = ""
        import re
        
        # 1. Clean <think> tags from the raw response first
        # This prevents the fallback from picking up reasoning text
        clean_response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
        
        # 2. Try Regex on cleaned response - harvest ALL blocks
        blocks = re.findall(r"```\s*python(.*?)```", clean_response, re.DOTALL | re.IGNORECASE)
        if not blocks:
            # Try generic blocks
            blocks = re.findall(r"```(.*?)```", clean_response, re.DOTALL)
            # Filter generic blocks to remove non-python (heuristic)
            blocks = [b[6:].strip() if b.strip().startswith("python") else b.strip() for b in blocks]
        
        if blocks:
            code_block = "\n\n".join(blocks)
        else:
            code_block = ""
        
        if not code_block:
             # Fallback: if no code blocks, but looks like code, use the cleaned text
             if "class " in clean_response and "nn.Module" in clean_response:
                 code_block = clean_response
             else:
                 return ExperimentConfig(algorithm="DCC", hyperparameters={}, env_id=task_id, domain="vision")


        return ExperimentConfig(
             algorithm="NAS", # Neural Architecture Search
             hyperparameters={
                 "model_code": code_block,
                 "lr": 1e-3,
                 "epochs": 5
             },
             env_id=task_id,
             domain="vision"
        ) 

