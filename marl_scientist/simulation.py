import asyncio
import time
import numpy as np
import random
from typing import Dict, List, Optional, Any
from marl_scientist.env.lab_env import LabEnvironment
from marl_scientist.agents.neural_researcher import NeuralResearcherAgent
from marl_scientist.safeguards.monitor import SingularityMonitor
from marl_scientist.utils.monitor import export_dashboard_data
from marl_scientist.utils.plotter import plot_training_curves
from marl_scientist.utils.logger import setup_logger
from marl_scientist.knowledge.real_store import RealKnowledgeStore
from marl_scientist.knowledge.watcher import KnowledgeWatcher
from marl_scientist.agents.memory import TrajectoryMemory
from marl_scientist.safeguards.symbolic_checker import SymbolicValidator

from marl_scientist.agents.trainer import MetaPPOTrainer

class Simulation:
    def __init__(self, steps: int, num_agents: int, max_agents: int, agent_type: str, visual: bool):
        self.steps = steps
        self.num_agents = num_agents
        self.max_agents = max_agents
        self.agent_type = agent_type
        self.visual = visual
        
        self.log = setup_logger("simulation.log")
        self.agents: Dict[str, NeuralResearcherAgent] = {}
        self.history: Dict[str, List[float]] = {}
        self.active_counts: Dict[str, int] = {}
        self.global_completions = 0
        self.population_reward_history = []
        self.reward_window = 20
        self.last_vision_time = time.time()
        
        # [PHASE 2] Training state
        self.training_buffer = []
        self.pending_train_data = {} # sub_id -> train_data_dict
        self.update_interval = 10 # Update brain every 10 experiments
        
        self.profiles = {
            0: np.array([0.8, 0.1, 0.1]), # The Perf-Maximizer
            1: np.array([0.1, 0.8, 0.1]), # The Fast-Efficient
            2: np.array([0.1, 0.1, 0.8]), # The Stable-Reliable
            3: np.array([0.4, 0.3, 0.3]), # The Balanced
        }
        self.profile_names = ["Perf-Max", "Fast-Efficient", "Stable-Reliable", "Balanced"]

    async def setup(self):
        self.log.info("[bold green]=== Initializing Async Meta-Scientist Ecosystem ===[/bold green]")
        
        benchmarks = ["CartPole-v1", "Acrobot-v1", "Pendulum-v1", "LunarLander-v3", "MountainCarContinuous-v0", "Hopper-v4", "Walker2d-v4", "HalfCheetah-v4"]
        self.lab = LabEnvironment(authorized_benchmarks=benchmarks, max_workers=self.max_agents, visual=self.visual)
        
        self.kb = RealKnowledgeStore()
        await self.kb.ingest_folder("knowledge/")
        
        self.watcher = KnowledgeWatcher(watch_dir="knowledge/")
        self.watcher.start()
        
        self.monitor = SingularityMonitor()
        self.tm = TrajectoryMemory(persistence_path="trajectories.pkl")
        self.validator = SymbolicValidator(time_budget_per_experiment=600.0)
        
        # [PHASE 2] Joint Brain Training?
        # In a real swarm, we might have one trainer per agent or a shared brain.
        # For simplicity, if they all share the same Brain architecture, we can update them similarly.
        # But each agent currently has its OWN Brain instance. 
        # Let's target the trainer to the first agent's brain as a representative, 
        # or just assume they are clones for now.
        
        for i in range(self.num_agents):
            self.spawn_agent(i + 1)
            
        # We'll use Agent_1 as the "Lead Scientist" whose brain we update and then sync.
        if "Agent_1" in self.agents and hasattr(self.agents["Agent_1"], "brain"):
            self.trainer = MetaPPOTrainer(self.agents["Agent_1"].brain)
        else:
            self.trainer = None

    def spawn_agent(self, idx: int):
        aid = f"Agent_{idx}"
        self.log.info(f"[Population] Spawning new agent: [bold cyan]{aid}[/bold cyan]")
        
        # In Phase 1, we still use NeuralResearcherAgent as it was.
        # Dependency injection will be improved in subsequent steps if needed, 
        # but let's follow the existing pattern for now within the Simulation class.
        shard = self.kb.get_shard(idx, 100)
        
        if self.agent_type == "neural":
            agent = NeuralResearcherAgent(
                agent_id=aid, 
                knowledge_store=shard, 
                trajectory_memory=self.tm, 
                validator=self.validator
            )
        else:
            from marl_scientist.agents.researcher import ResearcherAgent
            agent = ResearcherAgent(
                agent_id=aid, 
                knowledge_store=shard, 
                trajectory_memory=self.tm
            )
        
        profile_idx = (idx - 1) % len(self.profiles)
        if hasattr(agent, "preference_vec"):
            agent.preference_vec = self.profiles[profile_idx]
            self.log.info(f"  - Profile: [yellow]{self.profile_names[profile_idx]}[/yellow] {agent.preference_vec}")
        
        try:
            if hasattr(agent, "load"):
                agent.load(f"saves/{aid}.pkl")
        except:
            pass
        
        self.agents[aid] = agent
        self.active_counts[aid] = 0
        return agent

    async def run(self):
        await self.setup()
        self.log.info("Starting Async Event Loop...")
        
        try:
            while self.global_completions < self.steps:
                # A. Poll for Results
                await self.process_results()
                
                # B. Knowledge Updates
                await self.process_knowledge_updates()
                
                # C. Organic Scaling
                self.check_scaling()
                
                # D. Dispatch Proposals
                await self.dispatch_proposals()
                
                # E. Periodic Vision Analysis
                await self.periodic_vision_analysis()
                
                await asyncio.sleep(0.1)
                
        except asyncio.CancelledError:
            self.log.warning("Simulation task cancelled.")
        except Exception as e:
            self.log.error(f"[Simulation] Error in main loop: {e}", exc_info=True)
        finally:
            await self.cleanup()

    async def process_results(self):
        prefs = {aid: a.preference_vec for aid, a in self.agents.items()}
        # Lab poll is still synchronous in its implementation, but we call it here.
        # In a full async refactor, LabEnvironment would be async.
        results, rewards = self.lab.poll_results(agent_preferences=prefs)
        
        if not results:
            return
            
        for sub_id, res in results.items():
            aid = sub_id.split("_h")[0]
            self.log.info(f"[Event] Experiment completed for {sub_id} (Agent: {aid}). Reward: {res.final_mean_reward:.1f}")
            
            if res:
                if not res.config.hyperparameters.get("is_horizon"):
                    self.population_reward_history.append(rewards[sub_id])
                    if len(self.population_reward_history) > self.reward_window:
                        self.population_reward_history.pop(0)
                        
                    if aid not in self.history: self.history[aid] = []
                    self.history[aid].append(res.final_mean_reward)
                    
                    # [PHASE 2] Collect Training Transition
                    if sub_id in self.pending_train_data:
                        train_entry = self.pending_train_data.pop(sub_id)
                        train_entry["reward"] = rewards[sub_id]
                        train_entry["done"] = False # Single step episodic update
                        self.training_buffer.append(train_entry)
                        
                    self.global_completions += 1
                
                pop_mean = sum(self.population_reward_history) / len(self.population_reward_history) if self.population_reward_history else 0.0
                survival_cost = 0.05
                
                # [PHASE 4] Intrinsic Reward (Mastery Bonus)
                intrinsic_bonus = 0.0
                if aid in self.agents:
                    agent = self.agents[aid]
                    env_id = res.config.env_id
                    current_comp = agent.competency_scores.get(env_id, 0.0)
                    if res.final_mean_reward >= current_comp:
                        # Bonus for beating or matching best ever
                        intrinsic_bonus = 0.1
                
                relative_reward = (rewards[sub_id] - pop_mean) - survival_cost + intrinsic_bonus
                
                # Update the Reward in training buffer for the brain to see
                if not res.config.hyperparameters.get("is_horizon"):
                     # Re-find it in buffer (simple since it was just appended)
                     if self.training_buffer and self.training_buffer[-1]["reward"] == rewards[sub_id]:
                          self.training_buffer[-1]["reward"] = relative_reward

                self.log.info(f"  - Meta-Reward: {rewards[sub_id]:.2f} (Rel: {relative_reward:+.2f}, Bonus: {intrinsic_bonus})")
                
                if aid in self.agents:
                    await self.agents[aid].update_knowledge(res)
                    self.monitor.check_safety(res)
                
                # [REFINEMENT] Early Unlock for MuJoCo
                # If any agent cracks Pendulum (> -500), force unlock Tier 2
                if res.config.env_id == "Pendulum-v1" and res.final_mean_reward > -500.0:
                    if self.lab.tier < 2:
                        self.log.info(f"[Unlock] Agent {aid} cracked Pendulum ({res.final_mean_reward:.1f} > -500). Unlocking MuJoCo Tier!")
                        self.lab.tier = 2
            
            if aid in self.active_counts:
                self.active_counts[aid] -= 1
            
            self.log.info(f"--- [bold yellow]Progress: {self.global_completions}/{self.steps}[/bold yellow] (Best: {self.lab.best_reward:.1f}) ---")
            
            # [PHASE 2] Trigger Update
            if self.trainer and len(self.training_buffer) >= self.update_interval:
                self.log.info(f"[Brain] Triggering Meta-Brain Update with {len(self.training_buffer)} transitions...")
                loss, entropy = self.trainer.update(self.training_buffer)
                self.log.info(f"  - Meta-Loss: {loss:.44f}")
                self.log.info(f"  - Avg Entropy: {entropy:.4f}")
                self.training_buffer = []
                # Sync weights to all other agents
                if "Agent_1" in self.agents and hasattr(self.agents["Agent_1"], "brain"):
                    lead_brain = self.agents["Agent_1"].brain
                    for other_aid, other_agent in self.agents.items():
                        if other_aid != "Agent_1" and hasattr(other_agent, "harvest_weights"):
                            other_agent.harvest_weights(lead_brain, tau=0.8) # Strong sync

            # Harvesting Trigger
            if self.global_completions % 5 == 0 and len(self.agents) > 1:
                self.trigger_harvesting()
                
        export_dashboard_data(list(self.agents.values()), self.global_completions)

    def trigger_harvesting(self):
        # [NEW] Harvesting Limit
        if not hasattr(self, 'harvest_counts'):
            self.harvest_counts = {} # (source, target) -> count

        try:
            best_aid = max(self.agents.keys(), key=lambda x: self.history.get(x, [-float('inf')])[-1])
            worst_aid = min(self.agents.keys(), key=lambda x: self.history.get(x, [float('inf')])[-1])
            
            if best_aid != worst_aid:
                # Check limit
                pair_key = (best_aid, worst_aid)
                count = self.harvest_counts.get(pair_key, 0)
                if count >= 3:
                     self.log.info(f"[Competitive] Harvesting Limit reached for {worst_aid} -> {best_aid}. Skipping.")
                     return

                self.log.info(f"[Competitive] {worst_aid} is harvesting insights from {best_aid}...")
                best_agent = self.agents[best_aid]
                worst_agent = self.agents[worst_aid]
                if hasattr(best_agent, "brain") and hasattr(worst_agent, "harvest_weights"):
                    worst_agent.harvest_weights(best_agent.brain, tau=0.1)
                self.harvest_counts[pair_key] = count + 1
        except ValueError:
            pass

    async def process_knowledge_updates(self):
        new_files = self.watcher.get_new_files()
        if new_files:
            self.log.info(f"[Knowledge] Processing {len(new_files)} new papers...")
            await self.kb.process_file_queue(new_files)

    def check_scaling(self):
        idle_count = sum(1 for c in self.active_counts.values() if c == 0)
        busy_count = sum(c for c in self.active_counts.values())
        total_agents = len(self.agents)
        max_workers = self.lab.executor._max_workers
        
        if idle_count == 0 and busy_count < max_workers and total_agents < self.max_agents:
            self.log.info(f"[Scaling] All agents busy ({busy_count}/{max_workers}). Expansion triggered!")
            self.spawn_agent(total_agents + 1)

    async def dispatch_proposals(self):
        busy_count = sum(c for c in self.active_counts.values())
        max_workers = self.lab.executor._max_workers
        
        # [MULTI-TASK] Diversity Round Tracking
        if not hasattr(self, 'diversity_counter'):
             self.diversity_counter = 0
             
        # Increment counter roughly every "batch" of dispatches. 
        # Since this runs in a loop, we increment if we actually dispatch something or just periodically.
        # Let's count "dispatch events" roughly.
        self.diversity_counter += 1
        is_diversity_round = (self.diversity_counter % 1000 == 0) # Every ~100 seconds
        if is_diversity_round:
             self.log.info("[MultiTask] Diversity Round Active! Forcing environment mixing.")

        # [NEW] Check for System Change Proposals (Every 100 ticks or when significant)
        if self.diversity_counter % 100 == 0:
            for aid, agent in self.agents.items():
                prop = agent.propose_system_change()
                if prop:
                    # Only apply and log if it's a NEW value
                    current_val = getattr(self, prop["target"], None)
                    if current_val != prop["value"]:
                        self.log.info(f"[Self-Modification] Agent {aid} proposed system change: {prop['target']} -> {prop['value']} ({prop['reason']})")
                        # Apply change
                        if prop["target"] == "update_interval":
                            self.update_interval = prop["value"]
                        # Add more targets as needed

        proposal_buffer = []
        for aid, agent in self.agents.items():
            if self.active_counts[aid] == 0:
                recent_perf = self.history.get(aid, [0.0])[-1]
                priority = -recent_perf
                
                obs = self.lab.get_observation()
                # [PHASE 2] New return type
                configs, train_data = await agent.propose_experiment(obs)
                
                if isinstance(configs, list):
                    for i, cfg in enumerate(configs):
                        sid = f"{aid}_h{i}"
                        cfg.hyperparameters["sub_id"] = sid
                        proposal_buffer.append((priority, aid, cfg, train_data))
                else:
                    proposal_buffer.append((priority, aid, configs, train_data))
                
                self.active_counts[aid] = -1 # QUEUED
                
        proposal_buffer.sort(key=lambda x: x[0])
        elite_aids = [aid for _, aid, _, _ in proposal_buffer[:3]]
        
        for _, aid, config, train_data in proposal_buffer:
            if busy_count < max_workers:
                # Process proposal
                if not self.process_proposal(aid, config, elite_aids, is_diversity_round):
                    continue
                
                self.log.info(f"[Dispatch] {aid} -> {config.env_id} for {config.hyperparameters.get('total_timesteps', 0)//1000}k steps")
                
                # [LOGGING] Dump hyperparameters for correlation
                hp_log = {k: v for k, v in config.hyperparameters.items() if k not in ['sub_id', 'parent_agent']}
                self.log.info(f"  - Params: {hp_log}")

                sub_id = config.hyperparameters.get("sub_id", aid)
                # [PHASE 2] Store train data
                if train_data:
                    self.pending_train_data[sub_id] = train_data

                try:
                    self.lab.submit_experiment(sub_id, config)
                    if self.active_counts[aid] == -1: self.active_counts[aid] = 0
                    self.active_counts[aid] += 1
                    busy_count += 1
                except Exception as e:
                    self.log.error(f"[Dispatch] Error submitting experiment for {aid}: {e}")
                    self.active_counts[aid] = 0
                    if sub_id in self.pending_train_data:
                        del self.pending_train_data[sub_id]
            else:
                self.active_counts[aid] = 0

    def process_proposal(self, aid: str, config: Any, elite_aids: List[str], diversity_mode: bool = False) -> bool:
        unlocked_envs = self.lab.tiers[self.lab.tier]
        
        is_valid, violations = self.validator.validate(config)
        if not is_valid:
            self.log.warning(f"[SymbolicGuard] Proposal from {aid} REJECTED: {violations}")
            self.active_counts[aid] = 0
            return False

        # [MULTI-TASK] Environment Mixing
        # Top-3 Elites stay on hardest tasks (Pioneer Bias)
        if aid in elite_aids:
            if config.env_id not in unlocked_envs:
                old_env = config.env_id
                config.env_id = random.choice(unlocked_envs)
                self.log.info(f"[Pioneer] Mandatory Redirection for Top-3 Elite {aid} from {old_env} to {config.env_id}!")
        else:
            # Non-Elites: Mix tasks
            # 1. Diversity Round -> Force random unlocked env
            # 2. Random 30% chance -> Force random unlocked env (Exploration/Positive Signal)
            if diversity_mode or random.random() < 0.30:
                # Pick any unlocked environment (including easier ones from lower tiers)
                all_allowed = self.lab.allowed_envs
                old_env = config.env_id
                config.env_id = random.choice(all_allowed)
                if old_env != config.env_id:
                     self.log.info(f"[MultiTask] {aid} redirected {old_env} -> {config.env_id} for Generalization.")

        # Compatibility Checks (Algo vs Env)
        env_meta = self.lab.env_metadata.get(config.env_id, {})
        if env_meta.get("is_continuous") and config.algorithm == "DQN":
            config.algorithm = "SAC"
            self.log.info(f"  - Algorithm adjusted to {config.algorithm} for Continuous compatibility.")
        elif env_meta.get("is_discrete") and config.algorithm == "SAC":
            config.algorithm = "PPO"
            self.log.info(f"  - Algorithm adjusted to {config.algorithm} for Discrete compatibility.")

        # Step Budget
        steps_map = {
            "CartPole-v1": 50000, 
            "Pendulum-v1": 100000, 
            "Acrobot-v1": 100000,
            "LunarLander-v3": 200000,
            "MountainCarContinuous-v0": 300000,
            "Hopper-v4": 500000,
            "Walker2d-v4": 500000,
            "HalfCheetah-v4": 1000000
        }
        
        # [DYNAMIC BUDGET] MountainCar Optimization
        if config.env_id == "MountainCarContinuous-v0":
            # [OVERRIDE] Metric-driven switch to TD3 (more robust for this env)
            config.algorithm = "TD3"

            agent = self.agents.get(aid)
            mastery = agent.competency_scores.get(config.env_id, -100.0) if agent else -100.0
            
            # Boost budget for exploration
            if mastery < -10.0:
                config.hyperparameters["total_timesteps"] = 300000
                self.log.info(f"  - [DynamicBudget] Setting conservative budget (300k) for MountainCar + TD3 Override.")
            else:
                config.hyperparameters["total_timesteps"] = 500000
                self.log.info(f"  - [DynamicBudget] Setting robust budget (500k) for MountainCar + TD3 Override.")
            
            # Ensure off-policy params are efficient
            if "learning_starts" not in config.hyperparameters:
                 config.hyperparameters["learning_starts"] = 1000
        else:
            config.hyperparameters["total_timesteps"] = steps_map.get(config.env_id, 50000)
        
        # [EARLY STOPPING] Pass threshold
        config.hyperparameters["stagnation_threshold"] = 1.0 # Default improvement threshold
        
        if config.hyperparameters.get("is_horizon"):
            config.hyperparameters["total_timesteps"] = 5000
            self.log.info(f"  - [Horizon] Using reduced budget (5k steps) for dry-run.")
            
        return True

    async def periodic_vision_analysis(self):
        if self.visual and (time.time() - self.last_vision_time > 60):
            self.log.info("[Vision] requesting Snapshot...")
            plot_path = plot_training_curves(self.history)
            try:
                analysis = self.kb.client.analyze_image(
                    plot_path,
                    prompt="Analyze training progress. Who is winning? Any anomalies?"
                )
                self.log.info(f"\n[bold magenta]=== GLM-4v Report ===[/bold magenta]\n{analysis}\n")
            except:
                pass
            self.last_vision_time = time.time()

    async def cleanup(self):
        self.log.info("Cleaning up simulation...")
        if hasattr(self, 'lab'):
            self.lab.close()
        if hasattr(self, 'watcher'):
            self.watcher.stop()
        for a in self.agents.values():
            a.save(f"saves/{a.agent_id}.pkl")
        self.log.info("Shutdown complete.")
