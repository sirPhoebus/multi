
from typing import Tuple, Dict, Any, Optional
import traceback
from marl_scientist.core import ExperimentConfig, ExperimentResult
from marl_scientist.evaluation.sb3_runner import SB3ExperimentRunner
from marl_scientist.evaluation.coding_runner import CodingExperimentRunner
from marl_scientist.evaluation.vision_runner import VisionExperimentRunner

def run_experiment_task(config: ExperimentConfig, agent_id: str, visual: bool = False) -> Tuple[str, ExperimentResult, Optional[str]]:
    """
    Top-level function for multiprocessing. 
    Instantiates a fresh Runner (stateless) and executes.
    """
    try:
        # Create a fresh runner for this process
        if config.domain == "coding":
            runner = CodingExperimentRunner(task_id=config.env_id)
        elif config.domain == "vision":
            runner = VisionExperimentRunner(task_id=config.env_id)
        else:
            # Default to RL
            runner = SB3ExperimentRunner(benchmark_env_id=config.env_id)
            
        result = runner.run(config, visual=visual, agent_id=agent_id)
        return agent_id, result, None
    except Exception as e:
        # capturing traceback for debugging
        tb = traceback.format_exc()
        return agent_id, None, str(e) + "\n" + tb
