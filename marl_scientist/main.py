import argparse
import asyncio
import sys
from marl_scientist.simulation import Simulation

async def main():
    parser = argparse.ArgumentParser(description="Meta-RL Scientist Async Event Loop")
    parser.add_argument("--steps", type=int, default=5, help="Number of completed experiments to target")
    parser.add_argument("--num-agents", type=int, default=1, help="Initial number of agents")
    parser.add_argument("--max-agents", type=int, default=8, help="Maximum scaling limit")
    parser.add_argument("--agent-type", type=str, default="neural", choices=["heuristic", "neural"], help="Type of researcher agent")
    parser.add_argument("--visual", action="store_true", help="Enable visual reporting (captures environment snapshots)")
    args = parser.parse_args()

    sim = Simulation(
        steps=args.steps,
        num_agents=args.num_agents,
        max_agents=args.max_agents,
        agent_type=args.agent_type,
        visual=args.visual
    )
    
    await sim.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n[Main] Fatal Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
