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
    args = parser.parse_args()
    
    # Setup Logger
    log = setup_logger("simulation.log")
    
    log.info("[bold green]=== Initializing Meta-RL Scientist Lab ===[/bold green]")
    
    # 1. Setup Environment
    lab = LabEnvironment(authorized_benchmarks=["CartPole-v1"])
    
    # [NEW] Real Knowledge Base integration
    from marl_scientist.knowledge.real_store import RealKnowledgeStore
    kb = RealKnowledgeStore()
    kb.ingest_references("marl_scientist/ref.md") # Ingest ~100 papers
    
    # 2. Setup Agents
    # Dynamic creation
    agents = [
        ResearcherAgent(agent_id=f"Agent_{i+1}")
        for i in range(args.num_agents)
    ]
    
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
            
            # 2. Agent Action
            actions = {}
            for agent in agents:
                config = agent.propose_experiment(obs)
                actions[agent.agent_id] = config
                log.info(f"Agent {agent.agent_id} proposes: [yellow]{config.algorithm}[/yellow] with lr={config.hyperparameters.get('learning_rate', 'N/A'):.2e}")
                
            # 3. Environment Step
            log.info("Running experiments (this may take a moment)...")
            results, rewards = lab.step(actions)
            
            # 4. Learning & Safety Check
            for agent in agents:
                if agent.agent_id not in results: 
                    # E.g. agent failed experiment
                    continue
                    
                result = results[agent.agent_id]
                reward = rewards.get(agent.agent_id, 0.0)
                
                log.info(f"Result for {agent.agent_id}: Reward=[bold]{result.final_mean_reward:.2f}[/bold], NoveltyBonus={reward:.2f}")
                history[agent.agent_id].append(result.final_mean_reward)
                
                monitor.check_safety(result)
                agent.update_knowledge(result)
                
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
        
    # Save on completion
    for agent in agents:
         agent.save(f"saves/{agent.agent_id}.pkl")
         
    log.info("\n[bold green]=== Experiment Complete ===[/bold green]")
    
if __name__ == "__main__":
    main()
