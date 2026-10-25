# Proposed path to decentralized AMR coordination

The current implementation provides a deterministic transition model, safety checks,
an interchangeable evaluator, and replayable simulation traces. The remaining work
should be validated in small stages; the abstract's intended architecture is not yet
an implemented distributed controller.

## 1. Establish measurable planning baselines

Create fixed scenarios for obstacle detours, narrow corridors, competing service
locations, unreachable goals, different fleet sizes, and different block counts.
Compare completion rate, makespan, movement count, waiting, and planning latency.
Keep failed cases as benchmarks rather than removing them to improve reported results.

Add obstacle-aware shortest-path distances and a planner interface so greedy planning
and joint-action search use identical simulator rules. Keep legal-action enumeration
and transition validation authoritative. Demonstrate improvement on solvable detours
without regressing replay determinism or collision safety.

## 2. Separate proposals from agreement

Have each agent generate a local ranked action list (initially top three). Construct
candidate joint actions, reject conflicts using the exact safety rules, and select one
joint action for simultaneous execution. Measure candidate pruning against exhaustive
small-grid enumeration so local top-three pruning does not silently hide every solution.

## 3. Introduce peer-to-peer communication

Define versioned messages with agent IDs, timestep IDs, self-state, candidate proposals,
and a deterministic state digest. Reconstruct the shared snapshot at every peer.
Specify agreement, duplicate handling, timeout behavior, and safe waiting when snapshots
or decisions disagree. Test delay, reordering, missing messages, and agent failure in the
simulator before using separate processes or robots.

## 4. Add learned policy and value guidance

Generate cooperative self-play data with deterministic seeds and separate training,
validation, and held-out scenarios. Train policy/value estimates to guide candidate
ranking. Compare against the fixed baselines and report failures as well as successes.
Learned scores must never bypass exact legality checks.

## 5. Evaluate decentralized PUCT

Implement bounded-depth PUCT first in one process, then exchange branch statistics
between peers. Specify shared action encoding, visit accounting, synchronization, and
agreement on the final joint action. Compare search quality and message overhead with
the same compute and communication budgets.

## 6. Move toward robot integration

Add a robot adapter (for example, ROS), measured execution feedback, localization
uncertainty, and interruption/replanning. Discrete collision safety alone is insufficient
for physical motion: validate timing, footprints, and a separate emergency-stop path.

Each milestone should end with a reproducible demo, documented limits, and a tested
commit. Push completed checkpoints with their actual timestamps. GitHub currently
requires pull requests before changes can enter `main`.
