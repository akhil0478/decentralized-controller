from __future__ import annotations

from .simulator import run
from .state import make_initial_state
from .visualize import ascii_world


def main() -> None:
    state = make_initial_state()
    result = run(state)

    print("Decentralized Multi-Agent Planner — Day 1")
    print("=" * 54)
    print(f"Success: {result.success}")
    print(f"Timesteps: {len(result.steps)}")
    print(f"Final stack height: {result.final_state.stack_height}/{len(state.blocks)}")
    print()

    for step in result.steps:
        print(f"--- timestep {step.timestep} ---")
        print(ascii_world(step.before))
        print(f"Priority: {list(step.result.priority)}")
        for decision in step.result.decisions:
            print(f"A{decision.agent_id}: {decision.chosen}")
        print(f"Score: {step.result.score.as_tuple()} | accepted={step.result.accepted}")
        print()

    print("Final state:")
    print(ascii_world(result.final_state))


if __name__ == "__main__":
    main()
