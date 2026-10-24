# Decentralized Multi-Agent Planner

A deterministic simulation baseline for coordinating autonomous mobile robots (AMRs).
The default scenario uses three agents on a 7×7 grid to move four blocks from a source
to a destination stack. The simulator, safety rules, evaluator, and coordinator are
separate modules so future planning methods can be compared against the same rules.

## Run

Python 3.11 or newer is required. The text demo and core tests use only the standard library:

```bash
python3 -m src.planner.main --quiet
python3 scripts/run_demo.py --json
python3 -m unittest discover -s tests -v
```

The default scenario completes all four deliveries in 39 timesteps. Omitting `--quiet`
or `--json` prints each grid state, decision, priority order, and communication event.
The direct script also works when invoked by absolute path from another directory.

For animation and development checks, create an isolated environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m src.planner.main --animate --quiet
ruff check .
ruff format --check .
python -m unittest discover -s tests -v
```

Animation tests skip when Matplotlib is absent; CI installs it and runs them without a GUI.
Install `requirements.txt` instead if you need animation without the development tools.

## Scenarios and results

```bash
python3 -m src.planner.main --grid-size 10 --blocks 6 --max-steps 500 --json
```

`--json` emits a summary containing success, termination reason, attempted and executed
step counts, stack height, block count, grid size, and agent count. Exit codes are `0`
for completion, `1` for an unfinished simulation, and `2` for invalid command arguments.

The Python API also accepts custom agents, obstacles, source, and destination:

```python
from src.planner.simulator import run
from src.planner.state import make_initial_state

state = make_initial_state(
    agent_positions=((1, 1), (5, 5)),
    source=(1, 5),
    destination=(5, 1),
    obstacles=frozenset({(3, 3)}),
    grid_size=5,
    block_count=2,
)
result = run(state, max_steps=250)
print(result.success, result.termination_reason)
```

Grids must be at least 2×2, agent positions must be distinct, and block counts must be
nonnegative. Default source and destination are opposite corners; default agents use
the other corners and center, with duplicate positions omitted on very small grids.

Runs stop on `goal_reached`, `step_limit`, `rejected`, `stalled` (all agents wait), or
`cycle_detected` (a repeated task configuration). Repeat detection ignores time and
previous velocity. Time-dependent evaluators that intentionally revisit configurations
may therefore require a different stopping policy.

`result.steps` retains attempted decisions, including rejected proposals.
`result.states` contains only the initial state and executed transitions, and is the
trajectory used by animation. A rejection never changes the world. Alternative evaluators
can be passed as `run(state, evaluator=my_evaluator)` and receive `(state, movement_cost)`.

## Planning and safety rules

- Actions are up, down, left, right, wait, pickup, and drop. Movement is one cell per step.
- Pickup and drop require an adjacent cell; an agent can carry at most one block.
- Transitions reject boundary crossings, obstacles, cell collisions, simultaneous swaps,
  missing or extra agent actions, and competing pickups that exceed available stock.
- World states validate unique IDs, positions, reciprocal block ownership, and contiguous
  destination stack levels. Source and destination must be distinct and unobstructed.
- Simultaneous pickups assign the lowest available block IDs in agent-ID order; simultaneous
  drops assign stack levels in agent-ID order. Dictionary insertion order has no effect.
- Carriers receive priority, followed by distance to their target and then agent ID.
- The coordinator commits actions sequentially and conservatively holds uncommitted
  agents stationary. The exact transition model applies the resulting joint action simultaneously.
- Scores compare lexicographically: stack height, carried blocks, negative task distance,
  negative proximity risk, and negative movement cost. Logged tuples are authoritative;
  the optional weighted `Score.scalar()` display is not equivalent to that ordering.

## Current limits

This remains an **in-process coordination prototype**. A single coordinator implements
the priority sequence; state and decision broadcasts are log events, not peer-to-peer
network messages. It does not yet implement a decentralized consensus protocol.

The evaluator uses Manhattan distance and one-step greedy decisions. It can stall or
reject useful detours around obstacles and other robots; it does not guarantee task
completion for every solvable map or prevent every cooperative deadlock. Agents may
occupy source and destination cells because those cells are service locations, not obstacles.

Neural policies, self-play training, MCTS/PUCT, learned world models, physical robot
control, communication faults, and real-time sensing are not implemented.

## Development direction

The proposed path toward the decentralized Edge-AI objective is documented in
[docs/ROADMAP.md](docs/ROADMAP.md). The next milestone is a fixed scenario benchmark and
an obstacle-aware planning baseline, before introducing learned decisions or networking.

CI runs lint, formatting, unit and animation tests, and both demo entry points on
Python 3.11 and 3.14.
