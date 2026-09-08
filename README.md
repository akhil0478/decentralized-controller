# Decentralized Multi-Agent Planner — Day 1 Prototype

A small, deterministic prototype for the multi-agent planning architecture being investigated for SIH-style decentralized AMR coordination.

## Today's scope

- 7×7 grid
- 3 identical agents
- 4 blocks in a single source cell
- 1 block per stack layer, stacked vertically
- One-cell discrete movement
- Actions: up/down/left/right/wait/pickup/drop
- Pickup/drop only from an adjacent cell
- No shared cells and no simultaneous swapping
- Priority ordering based on carrying state and distance to the relevant target
- Sequential action commitment in priority order
- Global deterministic state evaluator
- Acceptance/rejection of the proposed next global state
- Full timestep logging
- ASCII and matplotlib visualization
- Unit tests

The neural evaluator, RL, MCTS, and learned world model are intentionally not included yet. The interfaces are designed so a learned evaluator can later replace the deterministic evaluator without rewriting the simulator.

## Mathematical core

Global state:

    S_t = (s_1, s_2, ..., s_N, B, L)

Transition:

    S_{t+1} = T(S_t, A_t)

Global evaluator:

    V(S) = lexicographic progress score

Agent decision:

    a_i* = argmax_a V(T(S, a, committed_higher_priority_actions))

## Run

```bash
python -m src.planner.main
```

Or:

```bash
python scripts/run_demo.py
```

## Tests

```bash
python -m unittest discover -s tests -v
```

## Design note

All runtime code is deterministic. Communication is simulated in-process: state broadcasts and decision broadcasts are represented explicitly in the event log, but there is no network dependency in this first prototype.
