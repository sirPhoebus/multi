
from typing import List, Dict, Any, Optional
import gymnasium as gym

class BaseDomain:
    def get_tasks(self, tier: int) -> List[str]:
        return []
    
    def get_metadata(self, task_id: str) -> Dict[str, Any]:
        return {}
    
    def check_promotion(self, task_id: str, score: float) -> bool:
        return False

class RLDomain(BaseDomain):
    def __init__(self):
        self.tiers = {
            0: ["CartPole-v1", "Pendulum-v1"],
            1: ["LunarLander-v3", "Acrobot-v1"],
            2: ["MountainCarContinuous-v0"],
            3: ["Hopper-v4", "Walker2d-v4", "HalfCheetah-v4"]
        }
    
    def get_tasks(self, tier: int) -> List[str]:
        tasks = []
        for t in range(tier + 1):
            tasks.extend(self.tiers.get(t, []))
        return tasks

    def get_metadata(self, task_id: str) -> Dict[str, Any]:
        try:
            temp_env = gym.make(task_id)
            meta = {
                "domain": "rl",
                "env_id": task_id,
                "is_discrete": isinstance(temp_env.action_space, gym.spaces.Discrete),
                "is_continuous": isinstance(temp_env.action_space, gym.spaces.Box),
            }
            temp_env.close()
            return meta
        except:
            return {"domain": "rl", "env_id": task_id, "error": "load_failed"}

class CodingDomain(BaseDomain):
    def __init__(self):
        self.tiers = {
            0: ["StringReverse"],
            1: ["ListSort"],
            2: ["Fibonacci"]
        }
        
    def get_tasks(self, tier: int) -> List[str]:
        # For now, coding tasks are always available or unlock similarly
        tasks = []
        for t in range(tier + 1):
            tasks.extend(self.tiers.get(t, []))
        return tasks
        
    def get_metadata(self, task_id: str) -> Dict[str, Any]:
        descriptions = {
            "StringReverse": "Write a function solution(x: str) -> str that reverses the string.",
            "ListSort": "Write a function solution(x: List[int]) -> List[int] that sorts the list.",
            "Fibonacci": "Write a function solution(n: int) -> int that returns the nth Fibonacci number."
        }
        return {
            "domain": "coding",
            "env_id": task_id,
            "description": descriptions.get(task_id, "Unknown Coding Task"),
            "language": "python"
        }


class VisionDomain(BaseDomain):
    def __init__(self):
        # We simulate split CIFAR-100 tasks
        # Task 0: Classes 0-4
        # Task 1: Classes 5-9 ...
        self.num_tasks = 20
        self.tiers = {
            0: [f"CIFAR100-Task-{i}" for i in range(5)],
            1: [f"CIFAR100-Task-{i}" for i in range(5, 10)],
            2: [f"CIFAR100-Task-{i}" for i in range(10, 15)],
            3: [f"CIFAR100-Task-{i}" for i in range(15, 20)]
        }

    def get_tasks(self, tier: int) -> List[str]:
        tasks = []
        for t in range(tier + 1):
            tasks.extend(self.tiers.get(t, []))
        return tasks

    def get_metadata(self, task_id: str) -> Dict[str, Any]:
        # Parse "CIFAR100-Task-X"
        try:
            tid = int(task_id.split("-")[-1])
        except:
            tid = 0
            
        return {
            "domain": "vision",
            "env_id": task_id,
            "task_index": tid,
            "input_dim": 3072, # 3*32*32
            "num_classes": 5,   # 5 classes per split
            "is_continuous": False
        }
