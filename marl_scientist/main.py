import argparse
import time
from marl_scientist.env.lab_env import LabEnvironment
from marl_scientist.agents.researcher import ResearcherAgent
from marl_scientist.safeguards.monitor import SingularityMonitor
from marl_scientist.utils.monitor import export_dashboard_data
from marl_scientist.utils.plotter import plot_training_curves
from marl_scientist.utils.logger import setup_logger

def main():
    parser = argparse.ArgumentParser(description="Meta-RL Scientist Main Loop")
    parser.add_argument("--steps", type=int, default=5, help="Number of meta-steps to run")
    parser.add_argument("--num-agents", type=int, default=2, help="Number of agents to simulate")
    parser.add_argument("--agent-type", type=str, default="heuristic", choices=["heuristic", "neural"], help="Type of researcher agent")
    args = parser.parse_args()
    
    # Setup Logger
    log = setup_logger("simulation.log")
    
    log.info("[bold green]=== Initializing Meta-RL Scientist Lab ===[/bold green]")
    
    # 1. Setup Environment
    lab = LabEnvironment(authorized_benchmarks=["CartPole-v1", "Acrobot-v1", "Pendulum-v1"])
    
    # [NEW] Real Knowledge Base integration
    from marl_scientist.knowledge.real_store import RealKnowledgeStore
    kb = RealKnowledgeStore()
    
    # Initial scan of existing files
    kb.ingest_folder("knowledge/") 
    
    # [NEW] Start Async Watcher
    from marl_scientist.knowledge.watcher import KnowledgeWatcher
    watcher = KnowledgeWatcher(watch_dir="knowledge/")
    watcher.start()
    
    # 2. Setup Agents
    from marl_scientist.agents.neural_researcher import NeuralResearcherAgent
    
    agents = []
    for i in range(args.num_agents):
        agent_id = f"Agent_{i+1}"
        if args.agent_type == "neural":
            agent = NeuralResearcherAgent(agent_id=agent_id)
        else:
            agent = ResearcherAgent(agent_id=agent_id)
        agents.append(agent)
    
    # [NEW] Load previous state if available
    for agent in agents:
        agent.load(f"saves/{agent.agent_id}.pkl")
    
    # [NEW] Distribute Knowledge Shards
    log.info("Distributing unique knowledge shards to agents...")
    for i, a in enumerate(agents):
        shard = kb.get_shard(i, len(agents))
        a.set_knowledge_store(shard)
    
    monitor = SingularityMonitor()
    
    # Track history for plotting
    history = {a.agent_id: [] for a in agents}
    
    log.info(f"Initialized {len(agents)} agents. Starting Meta-Loop for {args.steps} steps...")
    
    try:
        for step in range(args.steps):
            log.info(f"\n[bold cyan]--- Meta-Step {step+1} ---[/bold cyan]")
            
            # 1. Observation
            obs = lab.get_observation()
            
            # [NEW] Process any new knowledge files detected by the watcher
            new_files = watcher.get_new_files()
            if new_files:
                log.info(f"[KnowledgeWatcher] Detected {len(new_files)} new files. Processing...")
                kb.process_file_queue(new_files)
            
            # 2. Agent Action & Resource Allocation
            # Determine trials per agent based on performance
            # Default: 1 trial. Top performer: 2 trials (if safe).
            
            # Identify top agent
            best_agent = None
            if step > 0:
                best_agent = max(agents, key=lambda a: a.best_performance)
            
            trial_queue = []
            for agent in agents:
                trial_queue.append((agent, agent.agent_id)) # Standard slot
                
                # Bonus slot logic
                if step > 0 and agent == best_agent and agent.best_performance > 0:
                     log.info(f"[Resource] Granting BONUS trial to top agent: {agent.agent_id}")
                     trial_queue.append((agent, f"{agent.agent_id}_bonus"))

            actions = {}
            for agent_obj, run_id in trial_queue:
                # We need to distinguish the run_id in the actions dict
                # The agent itself doesn't know about run_id, so we just ask it to propose
                config = agent_obj.propose_experiment(obs)
                actions[run_id] = config
                log.info(f"Agent {run_id} proposes: [yellow]{config.algorithm}[/yellow] with lr={config.hyperparameters.get('learning_rate', 'N/A'):.2e}")
                
            # 3. Environment Step
            log.info("Running experiments (this may take a moment)...")
            results, rewards = lab.step(actions)
            
            # 4. Learning & Safety Check
            for agent_obj, run_id in trial_queue:
                if run_id not in results: 
                    # E.g. agent failed experiment
                    continue
                    
                result = results[run_id]
                if result is None:
                    log.warning(f"Experiment failed for {run_id}, skipping update.")
                    continue
                reward = rewards.get(run_id, 0.0)
                
                log.info(f"Result for {run_id}: Reward=[bold]{result.final_mean_reward:.2f}[/bold], NoveltyBonus={reward:.2f}")
                history[agent_obj.agent_id].append(result.final_mean_reward)
                
                monitor.check_safety(result)
                agent_obj.update_knowledge(result)
                # TODO: In PPO, we'd add this experience to the agent's buffer here.
                # Since agent_obj is the same instance, it learns from both trials!
                
            # 5. Dashboard Export
            export_dashboard_data(agents, step + 1)
            
            # 6. [NEW] Vision Analysis (Every 5 steps to avoid slowing down too much)
            if (step + 1) % 5 == 0:
                log.info("[Vision] Generating plot and requesting analysis (waiting for GLM-4v)...")
                plot_path = plot_training_curves(history)
                
                try:
                    analysis = kb.client.analyze_image(
                        plot_path, 
                        prompt="You are a Senior Data Scientist. Analyze this training curve. Are the agents improving? which one is better? Is there any instability?"
                    )
                    log.info(f"\n[bold magenta]=== GLM-4v Analysis ===[/bold magenta]\n{analysis}\n[bold magenta]=======================[/bold magenta]\n")
                except Exception as e:
                    log.error(f"[Vision Error] Analysis skipped: {e}")
                    
    except KeyboardInterrupt:
        log.warning("\n[!] Simulation interrupted by user. Exiting gracefully...")
        # Save on interrupt
        for agent in agents:
            agent.save(f"saves/{agent.agent_id}.pkl")
        
    finally:
        # Stop watcher
        watcher.stop()
        
    # Save on completion
    for agent in agents:
         agent.save(f"saves/{agent.agent_id}.pkl")
         
    log.info("\n[bold green]=== Experiment Complete ===[/bold green]")
    
if __name__ == "__main__":
    main()
