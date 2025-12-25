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
    lab = LabEnvironment(authorized_benchmarks=["CartPole-v1", "Acrobot-v1", "Pendulum-v1"])
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
                        # Update Agent
                        if aid in agents:
                            agents[aid].update_knowledge(res)
                            monitor.check_safety(res)
                            
                            if aid not in history: history[aid] = []
                            history[aid].append(res.final_mean_reward)
                    
                    # Mark Free
                    agent_status[aid] = "IDLE"
                    global_completions += 1
                
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
                
            # D. Job Submission
            for aid, agent in agents.items():
                if agent_status[aid] == "IDLE":
                    # Propose
                    obs = lab.get_observation()
                    config = agent.propose_experiment(obs)
                    
                    log.info(f"[Dispatch] {aid} -> {config.env_id} ({config.algorithm}) for 30k steps")
                    lab.submit_experiment(aid, config)
                    
                    agent_status[aid] = "BUSY"
            
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
