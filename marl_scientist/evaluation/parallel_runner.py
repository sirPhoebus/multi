
from typing import Tuple, Dict, Any, Optional
import traceback
from marl_scientist.core import ExperimentConfig, ExperimentResult
from marl_scientist.evaluation.sb3_runner import SB3ExperimentRunner

def run_experiment_task(config: ExperimentConfig, agent_id: str, visual: bool = False) -> Tuple[str, ExperimentResult, Optional[str]]:
    """
    Top-level function for multiprocessing. 
    Instantiates a fresh Runner (stateless) and executes.
    """
    try:
        # Create a fresh runner for this process
        # We use a default env for now, or could pass it in config if needed
        runner = SB3ExperimentRunner(benchmark_env_id=config.env_id)
        result = runner.run(config, visual=visual, agent_id=agent_id)
        return agent_id, result, None
    except Exception as e:
        # capturing traceback for debugging
        tb = traceback.format_exc()
        return agent_id, None, str(e) + "\n" + tb
