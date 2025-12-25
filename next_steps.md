Directives to Move Forward — Upgrading Our Brain

<findings>
The D3 Engine ("Deterministic-Deontic-Dynamic") is a neuro-symbolic runtime that strictly separates:

Probabilistic generation (neural/LLM components — creative proposal, hypothesis generation)
Deterministic state (symbolic layer — verification, de-duplication, safety enforcement)

Key innovations that directly address our swarm's current limitations:

Active Workspace vs Latent History
Only a tiny "active" context (volatile workspace) is kept in-token.
Everything else is compressed into Trajectory Vectors (high-dimensional embeddings of past states/decisions) stored in a vector DB.
This claims 99% compute reduction vs naive long-context LLMs — perfect for our agents that are starting to accumulate thousands of experiment histories.

Vector-Space De-duplication & Negative Knowledge Propagation
Failed trajectories are embedded and globally de-duplicated → the entire swarm instantly "learns" a failure mode without re-experiencing it.
This is huge for us: right now laggards harvest weights, but winners still occasionally repeat known-bad hyperparams.

Hierarchical Verification Stack
Multiple symbolic layers check proposals for validity before execution (e.g., type safety, resource bounds, causal consistency).
Prevents invalid experiments from ever hitting the lab — aligns perfectly with our need for safer self-modification.

Horizon Mode (from the companion paper)
Instead of single-agent long-chain reasoning, spawn a swarm of 10,000 lightweight agents exploring in parallel, then consolidate via "Flash-Gated Consensus".
Designed for open-ended engineering (exactly what our meta-scientists are doing).

Safety & Entropy Routing
Tasks are routed based on entropy: high-uncertainty → neural exploration; low-uncertainty → symbolic execution.
Built-in red-teaming and verification loops.
</findings>

Let's integrate D3 concepts incrementally without throwing away our working swarm. Here's a phased plan:
Phase 1: Immediate Low-Effort Wins (Next Commit)

Trajectory Vector Compression
Embed each completed experiment summary (config + outcome + time) into a fixed vector (use nomic-embed-text-v1.5 we already have). Store in our Chroma vector store with metadata (reward, time, success/fail).
Before proposing new experiments, retrieve top-k similar past trajectories → explicitly inject "negative knowledge" into the prompt ("Avoid these failed configs: ...").
Basic De-duplication
When an agent proposes a hyperparam set, compute its embedding and check cosine similarity > 0.95 against past experiments → reject if too similar and worse outcome.

Phase 2: Symbolic Verification Layer (1-2 Days)

Add a lightweight symbolic checker before dispatching to lab:
Validate hyperparams in sane ranges (e.g., lr between 1e-6 and 1e-1, batch_size power of 2, etc.)
Resource budgeting: estimate wall-time based on env + steps + parallel runners, reject if over agent's time budget.
Use simple Python asserts or even a tiny DSL for rules.


Phase 3: Active/Latent Split in Agent Brain (Major Upgrade)

Refactor agent history:
Active: last N experiments + current strategic intent.
Latent: everything else queried via vector retrieval.

Modify the LSTM brain input to concatenate retrieved trajectory vectors (projected to hidden size) with the sequence.

Phase 4: Horizon Mode Lite

When an agent enters "EXPLORE" with high uncertainty (e.g., new unlocked env), spawn 5-10 lightweight sub-agents with varied temperature/profiles, run short cheap simulations or dry-runs, then consensus-vote on the best proposal.

Phase 5: Full Safety Stack

Implement Flash-Gated Consensus for critical actions (self-modification attempts).
Add a global "red-team" agent that periodically challenges high-reward proposals.