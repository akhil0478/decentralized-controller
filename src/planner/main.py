from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from .constants import BLOCK_COUNT, GRID_SIZE
from .simulator import SimulationResult, run
from .state import make_initial_state
from .visualize import animate, ascii_world


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the deterministic multi-agent planner.")
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--grid-size", type=int, default=GRID_SIZE)
    parser.add_argument("--blocks", type=int, default=BLOCK_COUNT)
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--quiet", action="store_true", help="Print only the summary.")
    output.add_argument("--json", action="store_true", help="Print a machine-readable summary.")
    parser.add_argument(
        "--animate", action="store_true", help="Show the executed trajectory with Matplotlib."
    )
    args = parser.parse_args(argv)
    try:
        state = make_initial_state(grid_size=args.grid_size, block_count=args.blocks)
        result = run(state, max_steps=args.max_steps)
    except ValueError as exc:
        parser.error(str(exc))

    if args.json:
        print(
            json.dumps(
                {
                    "success": result.success,
                    "termination_reason": result.termination_reason,
                    "attempted_steps": len(result.steps),
                    "executed_steps": len(result.states) - 1,
                    "stack_height": result.final_state.stack_height,
                    "block_count": len(state.blocks),
                    "grid_size": state.grid_size,
                    "agent_count": len(state.agents),
                },
                sort_keys=True,
            )
        )
    else:
        print("Decentralized Multi-Agent Planner - deterministic baseline")
        print("=" * 54)
        print(f"Success: {result.success}")
        print(f"Termination: {result.termination_reason}")
        print(f"Attempted steps: {len(result.steps)}")
        print(f"Executed timesteps: {len(result.states) - 1}")
        print(f"Final stack height: {result.final_state.stack_height}/{len(state.blocks)}")
        if not args.quiet:
            _print_steps(result)

    if args.animate:
        animate(result)
    return 0 if result.success else 1


def _print_steps(result: SimulationResult) -> None:
    print()

    for step in result.steps:
        print(f"--- timestep {step.timestep} ---")
        print(ascii_world(step.before))
        print(f"Priority: {list(step.result.priority)}")
        for decision in step.result.decisions:
            print(f"A{decision.agent_id}: {decision.chosen}")
        print(f"Score: {step.result.score.as_tuple()} | accepted={step.result.accepted}")
        for event in step.result.communication_events:
            print(event)
        if step.result.rejection_reason:
            print(f"Rejection: {step.result.rejection_reason}")
        print()

    print("Final state:")
    print(ascii_world(result.final_state))


if __name__ == "__main__":
    raise SystemExit(main())
