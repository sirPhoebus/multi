
from typing import Dict, Any, List
import time
import dataclasses
from marl_scientist.core import ExperimentConfig, ExperimentResult

class CodingExperimentRunner:
    """
    Evaluates coding 'experiments' (submitted code solutions).
    """
    def __init__(self, task_id: str):
        self.task_id = task_id
        
    def run(self, config: ExperimentConfig, visual: bool = False, agent_id: str = "unknown") -> ExperimentResult:
        """
        Executes the submitted code against the task's test cases.
        """
        code = config.hyperparameters.get("code", "")
        start_time = time.time()
        
        # 1. Define Test Cases
        test_cases = self._get_test_cases(self.task_id)
        
        if not test_cases:
             return self._failure_result(config, "Unknown Task ID", start_time)
             
        # 2. Execute Code
        # WARNING: unsafe exec for demo purposes. In prod, use docker/sandbox.
        # We expect the code to define a function named 'solution'.
        
        env = {}
        try:
            exec(code, {}, env)
            if "solution" not in env:
                return self._failure_result(config, "Code did not define 'solution(x)' function.", start_time)
            
            solution_fn = env["solution"]
            
            # 3. Run Tests
            correct = 0
            total = len(test_cases)
            
            for inp, expected in test_cases:
                try:
                    out = solution_fn(inp)
                    if out == expected:
                        correct += 1
                except Exception as e:
                    pass # Fail this case
            
            success_rate = correct / total
            duration = time.time() - start_time
            
            # 4. Construct Result
            return ExperimentResult(
                config=config,
                final_mean_reward=success_rate * 100.0, # Scale to 0-100 to match RL somewhat
                training_curve=[success_rate * 100.0] * 10, # Flat curve for one-shot
                metrics={
                    "accuracy": success_rate,
                    "std_reward": 0.0,
                    "test_cases_passed": float(correct)
                },
                duration_seconds=duration,
                start_time=start_time,
                total_env_steps=len(test_cases),
                performance_score=success_rate,
                efficiency_score=1.0 if success_rate == 1.0 else 0.0,
                stability_score=1.0
            )

        except Exception as e:
            return self._failure_result(config, f"Execution Error: {str(e)}", start_time)

    def _get_test_cases(self, task_id):
        if task_id == "StringReverse":
            return [("hello", "olleh"), ("abc", "cba"), ("", ""), ("123", "321")]
        elif task_id == "ListSort":
            return [([3,1,2], [1,2,3]), ([], []), ([5], [5]), ([1,1,1], [1,1,1])]
        elif task_id == "Fibonacci":
            # 0-indexed: 0, 1, 1, 2, 3, 5...
            return [(0, 0), (1, 1), (5, 5), (6, 8), (10, 55)]
        return []

    def _failure_result(self, config, error_msg, start_time):
        return ExperimentResult(
            config=config,
            final_mean_reward=-10.0,
            training_curve=[],
            metrics={"error": 1.0},
            duration_seconds=time.time() - start_time,
            start_time=start_time,
            info={"error": error_msg}
        )
