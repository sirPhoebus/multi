
import gymnasium as gym
import numpy as np
from marl_scientist.agents.researcher import ResearcherAgent
from marl_scientist.evaluation.sb3_runner import SB3ExperimentRunner
from marl_scientist.core import ExperimentConfig

def test_config_generation_and_execution():
    print("Testing Config Generation and Execution...")
    
    agent = ResearcherAgent("TestAgent")
    runner = SB3ExperimentRunner("CartPole-v1")
    
    # Generate 10 random configs
    for i in range(10):
        print(f"\n--- Test {i+1} ---")
        config = agent._random_config()
        print(f"Algorithm: {config.algorithm}")
        print(f"Hyperparams: {config.hyperparameters}")
        
        try:
            # We don't want to run full training, just instantiation
            # So we Mock the gym environment or just run with extremely low budget if possible?
            # Actually, let's just instantiate the model using the runner logic
            # Use `run` but mocked? Runner run method creates env and model.
            
            # Let's inspect passing logic manually or just dry run
            # To dry run: we can check if it initializes without error.
            print("Initializing runner...")
            
            # This will fail if parameters are invalid
            # Using a trick: we want to only init, not learn.
            # But the runner.run() does both.
            # Let's trust the runner's instantiation part.
            # We call runner.run() but we might need to interrupt it or verify it works?
            # Actually, let's create a minimal test that calls runner._get_algo_class and inits it.
            
            algo_class = runner._get_algo_class(config.algorithm)
            
            # Reconstruct logic from runner to test instantiation
            env = gym.make("CartPole-v1")
            
            hp = config.hyperparameters.copy()
            policy_kwargs = {}
            if "net_arch" in hp: policy_kwargs["net_arch"] = hp.pop("net_arch")
            
            if "activation_fn" in hp:
                 # Simplified mapping for test
                 import torch.nn as nn
                 hp.pop("activation_fn") 
                 policy_kwargs["activation_fn"] = nn.ReLU # Dummy
            
            # Filter kwargs as done in the code (Simplified for test)
            algo_kwargs = {}
            for k, v in hp.items():
                if k in ["learning_rate", "gamma", "gae_lambda", "ent_coef", "vf_coef", "max_grad_norm", "n_steps", 
                         "buffer_size", "learning_starts", "batch_size", "tau", "train_freq", "gradient_steps",
                         "n_epochs", "clip_range", "target_kl",
                         "target_update_interval", "exploration_fraction", "exploration_final_eps"]:
                     # Note: real runner has precise filtering, here we just check if it crashes SB3
                     # SB3 will error if we pass "gae_lambda" to DQN
                     pass
                     
            # Actually running the REAL runner.run is better but time consuming.
            # Let's just run it! 
            # But we need to patch model.learn to be instant.
            
            import unittest.mock as mock
            with mock.patch(f"stable_baselines3.{config.algorithm}.learn") as mock_learn:
                 runner.run(config)
                 print("Instantiation Successful.")
                 
        except Exception as e:
            print(f"[FAIL] Config caused error: {e}")
            raise e

    print("\n[PASS] All 10 random diverse configs initialized successfully.")

def test_mutation():
    print("\nTesting Mutation...")
    agent = ResearcherAgent("TestAgent")
    base = agent._random_config()
    print(f"Base: {base}")
    mutated = agent._mutate_config(base)
    print(f"Mutated: {mutated}")
    assert mutated != base
    print("[PASS] Mutation works.")

if __name__ == "__main__":
    test_mutation()
    test_config_generation_and_execution()
