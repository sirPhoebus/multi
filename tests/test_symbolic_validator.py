from marl_scientist.core import ExperimentConfig
from marl_scientist.safeguards.symbolic_checker import SymbolicValidator

def test_symbolic_validator():
    print("Testing Symbolic Validator...")
    validator = SymbolicValidator(time_budget_per_experiment=100.0)
    
    # 1. Test Valid Config
    valid_config = ExperimentConfig(
        algorithm="PPO",
        hyperparameters={"learning_rate": 0.0003, "batch_size": 64, "total_timesteps": 10000},
        env_id="CartPole-v1"
    )
    is_valid, violations = validator.validate(valid_config)
    print(f"Valid Config: {is_valid}, Violations: {violations}")
    assert is_valid
    
    # 2. Test Invalid HP (Learning Rate)
    invalid_lr_config = ExperimentConfig(
        algorithm="PPO",
        hyperparameters={"learning_rate": 0.5, "batch_size": 64, "total_timesteps": 10000},
        env_id="CartPole-v1"
    )
    is_valid, violations = validator.validate(invalid_lr_config)
    print(f"Invalid LR: {is_valid}, Violations: {violations}")
    assert not is_valid
    assert any("learning_rate" in v for v in violations)
    
    # 3. Test Invalid Batch Size (Not power of 2)
    invalid_bs_config = ExperimentConfig(
        algorithm="PPO",
        hyperparameters={"learning_rate": 0.0003, "batch_size": 13, "total_timesteps": 10000},
        env_id="CartPole-v1"
    )
    is_valid, violations = validator.validate(invalid_bs_config)
    print(f"Invalid BS: {is_valid}, Violations: {violations}")
    assert not is_valid
    assert any("batch_size" in v or "power of 2" in v for v in violations)
    
    # 4. Test Resource Budget (Too many steps)
    over_budget_config = ExperimentConfig(
        algorithm="PPO",
        hyperparameters={"learning_rate": 0.0003, "batch_size": 64, "total_timesteps": 1000000},
        env_id="CartPole-v1"
    )
    is_valid, violations = validator.validate(over_budget_config)
    print(f"Over Budget: {is_valid}, Violations: {violations}")
    assert not is_valid
    assert any("exceeds budget" in v for v in violations)

    print("\nSymbolic Validator Tests Passed!")

if __name__ == "__main__":
    test_symbolic_validator()
