Double down on verifiable rewards: Make more parts of the lab auto-checkable (e.g., not just final reward, but intermediate metrics like stability, convergence speed).
Add explicit "reasoning trace" logging in agent proposals — force the LLM to output step-by-step justification, then reward based on whether the trace matches successful outcomes.
Embrace the jaggedness: Add a "domain entropy estimator" (like D3's routing) — if an env has high variance in recent rewards, route to neural exploration (higher temp, more scouts); if low, go symbolic/deterministic.
Ditch reliance on fixed benchmarks — our curriculum is already adaptive, keep pushing that.
Long-term: Add multimodal output — agents generate diagrams of hyperparam landscapes or training curves.
Short-term: Make the knowledge inbox support dropping images/PDFs (papers with figures) → embed via multimodal model.