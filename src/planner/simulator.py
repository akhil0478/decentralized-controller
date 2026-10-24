from __future__ import annotations

from dataclasses import dataclass

from .coordinator import CoordinationResult, coordinate_once
from .evaluator import evaluate
from .state import WorldState


@dataclass(frozen=True)
class StepLog:
    timestep: int
    before: WorldState
    result: CoordinationResult


@dataclass(frozen=True)
class SimulationResult:
    steps: tuple[StepLog, ...]
    final_state: WorldState
    success: bool


def run(initial_state: WorldState, max_steps: int = 250) -> SimulationResult:
    state = initial_state
    logs: list[StepLog] = []

    for _ in range(max_steps):
        if state.goal_reached:
            return SimulationResult(tuple(logs), state, True)

        result = coordinate_once(state, evaluator=evaluate)

        if not result.accepted:
            # Keep the proposal in the log, but do not execute a rejected action.
            logs.append(StepLog(state.timestep, state, result))
            return SimulationResult(tuple(logs), state, False)

        logs.append(StepLog(state.timestep, state, result))
        state = result.next_state

    return SimulationResult(tuple(logs), state, state.goal_reached)
