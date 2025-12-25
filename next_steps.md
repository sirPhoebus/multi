### __Critical Gaps & Improvement Areas__

#### __1. The "Meta-RL" Aspect is Currently Missing__

- __Current State__: Agents use simple random mutation (epsilon-greedy), not actual reinforcement learning.
- __Required Improvement__: Implement proper meta-RL where agents learn to propose better experiments over time. The `ResearcherAgent` should be an RNN policy trained via PPO/A2C on the meta-environment rewards.

#### __2. Action Space Limitations__

- __Current State__: Only 3 hyperparameters (`learning_rate`, `gamma`, `ent_coef`) for a single algorithm.

- __Required Improvement__: Expand to:

  - Multiple algorithms (PPO, A2C, DQN, SAC)
  - Architectural choices (network layers, activation functions)
  - Loss function modifications
  - Exploration strategy parameters

#### __3. Observation Space is Information-Poor__

- __Current State__: Only raw experiment history.

- __Required Improvement__: Include:

  - Causal graph embeddings
  - Performance trends (derivatives)
  - Novelty landscape features
  - Knowledge base summaries

#### __4. Causal Discovery is Stubbed__

- __Current State__: Empty `CausalGraph` class.
- __Required Improvement__: Implement causal inference (e.g., PC algorithm, neural causal models) to learn relationships like `ent_coef → exploration → final_reward`.

#### __5. Knowledge Integration (DONE)__

- __Status__: Implemented RAG workflow with `RealKnowledgeStore`.
- __Mechanism__: Agents query KB, LLM translates papers to JSON configs.

#### __6. Parallelization & Efficiency (DONE)__

- __Status__: Implemented `ProcessPoolExecutor` in `LabEnvironment`.
- __Performance__: Experiments run concurrently on available cores.

#### __7. Reward Function Design (DONE)__

- __Status__: Implemented Adaptive Meta-Reward.
- __Mechanism__: Weights shift from Novelty to Performance; Stability penalty included.

#### __8. Missing Meta-Training Loop__

- __Current State__: Main loop runs fixed steps without updating agent policies.
- __Required Improvement__: Implement outer-loop optimization that updates researcher agents based on cumulative meta-rewards.

#### __9. Benchmark Diversity__

- __Current State__: Only CartPole-v1.
- __Required Improvement__: Add benchmark suites (Classic Control, Atari, MuJoCo) to test generalization.
