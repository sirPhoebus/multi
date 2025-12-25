import argparse
import time
import numpy as np
import psutil
from marl_scientist.env.lab_env import LabEnvironment
from marl_scientist.agents.researcher import ResearcherAgent
from marl_scientist.agents.neural_researcher import NeuralResearcherAgent
from marl_scientist.safeguards.monitor import SingularityMonitor
from marl_scientist.utils.monitor import export_dashboard_data
from marl_scientist.utils.plotter import plot_training_curves
from marl_scientist.utils.logger import setup_logger
from marl_scientist.knowledge.real_store import RealKnowledgeStore
from marl_scientist.knowledge.watcher import KnowledgeWatcher

def main():
    parser = argparse.ArgumentParser(description="Meta-RL Scientist Async Event Loop")
    parser.add_argument("--steps", type=int, default=5, help="Number of completed experiments to target")
    parser.add_argument("--num-agents", type=int, default=1, help="Initial number of agents")
    parser.add_argument("--max-agents", type=int, default=8, help="Maximum scaling limit")
    parser.add_argument("--agent-type", type=str, default="heuristic", choices=["heuristic", "neural"], help="Type of researcher agent")
    args = parser.parse_args()
    
    log = setup_logger("simulation.log")
    log.info("[bold green]=== Initializing Async Meta-Scientist Ecosystem ===[/bold green]")
    
    # 1. Setup Environment
    benchmarks = ["CartPole-v1", "Acrobot-v1", "Pendulum-v1", "LunarLander-v3", "MountainCarContinuous-v0", "Hopper-v4", "Walker2d-v4", "HalfCheetah-v4"]
    lab = LabEnvironment(authorized_benchmarks=benchmarks, max_workers=args.max_agents)
    kb = RealKnowledgeStore()
    kb.ingest_folder("knowledge/") 
    
    watcher = KnowledgeWatcher(watch_dir="knowledge/")
    watcher.start()
    
    monitor = SingularityMonitor()
    
    # 2. Agent Population Management
    agents = {} # agent_id -> AgentObj
    agent_status = {} # agent_id -> "IDLE" or "BUSY"
    
    # [NEW] Scientist Profiles [Performance, Efficiency, Stability]
    PROFILES = {
        0: np.array([0.8, 0.1, 0.1]), # The Perf-Maximizer
        1: np.array([0.1, 0.8, 0.1]), # The Fast-Efficient
        2: np.array([0.1, 0.1, 0.8]), # The Stable-Reliable
        3: np.array([0.4, 0.3, 0.3]), # The Balanced
    }
    
    def spawn_agent(idx):
        aid = f"Agent_{idx}"
        log.info(f"[Population] Spawning new agent: [bold cyan]{aid}[/bold cyan]")
        agent = NeuralResearcherAgent(agent_id=aid)
        
        # Assign Profile
        profile_idx = idx % len(PROFILES)
        agent.preference_vec = PROFILES[profile_idx]
        profile_names = ["Perf-Max", "Fast-Efficient", "Stable-Reliable", "Balanced"]
        log.info(f"  - Profile: [yellow]{profile_names[profile_idx]}[/yellow] {agent.preference_vec}")

        # Try load state if exists
        try:
            agent.load(f"saves/{aid}.pkl")
        except:
            pass
        # Give knowledge
        shard = kb.get_shard(idx, 100) # Pseudo-shard
        agent.set_knowledge_store(shard)
        agents[aid] = agent
        agent_status[aid] = "IDLE"
        return agent

    # Initialize seed population
    for i in range(args.num_agents):
        spawn_agent(i+1)
        
    history = {} # agent_id -> [rewards]
    
    global_completions = 0
    start_time = time.time()
    last_vision_time = time.time()
    
    # [NEW] Competitive Dynamics
    population_reward_history = [] 
    reward_window = 20 # Mean over last 20 experiments
    
    try:
        log.info("Starting Event Loop...")
        
        while global_completions < args.steps:
            # A. Poll for Results
            # Pass Current Preferences for Meta-Reward calculation
            prefs = {aid: a.preference_vec for aid, a in agents.items()}
            results, rewards = lab.poll_results(agent_preferences=prefs)
            
            if results:
                for aid, res in results.items():
                    log.info(f"[Event] Experiment completed for {aid}. Reward: {res.final_mean_reward:.1f}")
                    
                    if res:
                        # [NEW] Relative Scaling Logic
                        population_reward_history.append(rewards[aid])
                        if len(population_reward_history) > reward_window:
                            population_reward_history.pop(0)
                        
                        pop_mean = sum(population_reward_history) / len(population_reward_history)
                        
                        # [NEW] Swarm Urgency (Survival Cost)
                        # As the simulation progresses, meta-rewards face a small decay.
                        # This increases the relative pressure for improvement.
                        survival_cost = 0.05
                        relative_reward = (rewards[aid] - pop_mean) - survival_cost
                        
                        log.info(f"  - Meta-Reward: {rewards[aid]:.2f} (Rel: {relative_reward:+.2f}, Cost: {survival_cost})")
                        
                        # Update Agent
                        if aid in agents:
                            # We update with RELATIVE reward to drive competition
                            agents[aid].update_knowledge(res)
                            monitor.check_safety(res)
                            
                            if aid not in history: history[aid] = []
                            history[aid].append(res.final_mean_reward)
                    
                    # Mark Free
                    agent_status[aid] = "IDLE"
                    global_completions += 1
                    
                    # Log Progress
                    log.info(f"--- [bold yellow]Progress: {global_completions}/{args.steps}[/bold yellow] (Best: {lab.best_reward:.1f}) ---")
                    
                    # [NEW] Weight Harvesting Trigger
                    if global_completions % 5 == 0 and len(agents) > 1:
                        # Find Global Best Agent (by recent mean)
                        best_aid = max(agents.keys(), key=lambda x: history.get(x, [-float('inf')])[-1])
                        # Find Worst Agent
                        worst_aid = min(agents.keys(), key=lambda x: history.get(x, [float('inf')])[-1])
                        
                        if best_aid != worst_aid:
                            log.info(f"[Competitive] {worst_aid} is harvesting insights from {best_aid}...")
                            agents[worst_aid].harvest_weights(agents[best_aid].brain, tau=0.1)
                
                # Update Dashboard immediately on new data
                export_dashboard_data(list(agents.values()), global_completions)

            # B. Knowledge Updates
            new_files = watcher.get_new_files()
            if new_files:
                log.info(f"[Knowledge] Processing {len(new_files)} new papers...")
                kb.process_file_queue(new_files)

            # C. Organic Scaling Logic
            # Count busy agents
            busy_count = sum(1 for s in agent_status.values() if s == "BUSY")
            idle_count = sum(1 for s in agent_status.values() if s == "IDLE")
            total_agents = len(agents)
            
            # Simple Heuristic: If everyone is busy, and we have CPU room, spawn more.
            # We assume lab.executor._max_workers is the limit.
            max_workers = lab.executor._max_workers
            
            if idle_count == 0 and busy_count < max_workers and total_agents < args.max_agents:
                # Check cooling - don't spawn too fast? No, let's just spawn if slot open.
                log.info(f"[Scaling] All agents busy ({busy_count}/{max_workers}). Expansion triggered!")
                spawn_agent(total_agents + 1)
                
            # C. Organic Scaling Logic
            # ... (Scaling logic stays same, checking available slots)
            
            # [NEW] Priority Scheduling Logic
            # We only submit if Lab has slots
            busy_count = sum(1 for s in agent_status.values() if s == "BUSY")
            max_workers = lab.executor._max_workers
            
            # 1. Collect Proposals from IDLE agents into a priority buffer
            proposal_buffer = [] # (priority, aid, config)
            for aid, agent in agents.items():
                if agent_status[aid] == "IDLE":
                    # Priority = -RecentMeanReward (higher reward -> lower priority value -> schedules first)
                    recent_perf = history.get(aid, [0.0])[-1]
                    priority = -recent_perf 
                    
                    obs = lab.get_observation()
                    config = agent.propose_experiment(obs)
                    proposal_buffer.append((priority, aid, config))
                    agent_status[aid] = "QUEUED" # Intermediate state
            
            # Sort buffer by priority
            proposal_buffer.sort(key=lambda x: x[0])
            
            # 2. Dispatch from buffer to Lab as long as slots are open
            n_elites = max(1, len(agents) // 4)
            elite_aids = [aid for _, aid, _ in proposal_buffer[:n_elites]]
            
            for _, aid, config in proposal_buffer:
                if busy_count < max_workers:
                    # [NEW] Pioneer Pressure
                    # If elite, we check if they are proposing from the highest unlocked tier
                    unlocked_envs = lab.tiers[lab.tier]
                    if aid in elite_aids and config.env_id not in unlocked_envs and lab.tier > 0:
                        # Force them to a random env from the new tier
                        import random
                        old_env = config.env_id
                        config.env_id = random.choice(unlocked_envs)
                        log.info(f"[Pioneer] Redirecting Elite {aid} from {old_env} to {config.env_id}!")

                    # [NEW] Tiered Step Budget
                    if config.env_id in ["CartPole-v1"]:
                        steps = 30000
                    elif config.env_id in ["Pendulum-v1", "Acrobot-v1"]:
                        steps = 100000
                    elif config.env_id in ["LunarLander-v3"]:
                        steps = 200000
                    elif config.env_id in ["MountainCarContinuous-v0"]:
                        steps = 300000
                    elif config.env_id in ["Hopper-v4", "Walker2d-v4"]:
                        steps = 500000
                    elif config.env_id in ["HalfCheetah-v4"]:
                        steps = 1000000
                    else:
                        steps = 50000
                        
                    config.hyperparameters["total_timesteps"] = steps
                    
                    log.info(f"[Dispatch] {aid} (Prio: {-priority:.1f}) -> {config.env_id} for {steps//1000}k steps")
                    lab.submit_experiment(aid, config)
                    agent_status[aid] = "BUSY"
                    busy_count += 1
                else:
                    # Put back to IDLE so we can re-evaluate priority next tick
                    agent_status[aid] = "IDLE"
            
            # ... (Periodic Vision Analysis logic stays same)
            
            # E. Periodic Vision Analysis (every 60s)
            if time.time() - last_vision_time > 60:
                log.info("[Vision] requesting Snapshot...")
                plot_path = plot_training_curves(history)
                try:
                    analysis = kb.client.analyze_image(
                        plot_path, 
                        prompt="Analyze training progress. Who is winning? Any anomalies?"
                    )
                    log.info(f"\n[bold magenta]=== GLM-4v Report ===[/bold magenta]\n{analysis}\n")
                except: pass
                last_vision_time = time.time()
                
            # Prevent CPU burn
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        log.warning("User Interrupt.")
    finally:
        lab.close()
        watcher.stop()
        for a in agents.values():
            a.save(f"saves/{a.agent_id}.pkl")
        log.info("Shutdown complete.")
    
    import sys
    sys.exit(0)

if __name__ == "__main__":
    main()
