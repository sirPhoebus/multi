from marl_scientist.core import ExperimentResult

class SingularityMonitor:
    """
    Monitors the experiment loop for signs of:
    1. Reward Hacking (unreasonably high rewards).
    2. Zero/NaN Loss (optimization failure).
    3. Resource starvation (not implemented yet).
    """
    
    def __init__(self, high_reward_threshold: float = 495.0): # Near perfect score for CartPole
        self.threshold = high_reward_threshold
        self.suspicious_count = 0
        
    def check_safety(self, result: ExperimentResult) -> bool:
        """Returns True if safe, False if intervention needed."""
        
        # Check for Reward Hacking / Solving "Too Fast"
        # If an agent consistently gets perfect scores immediately, it might be cheating (or just good).
        # We flag it.
        if result.final_mean_reward >= self.threshold:
            print(f"[MONITOR] High reward detected: {result.final_mean_reward}. Verifying...")
            self.suspicious_count += 1
            
        # Check for numeric instability
        # (Assuming we pass loss in result metrics in future)
        
        return True # For now, just logging, no shutdown.
