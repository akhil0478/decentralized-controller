from __future__ import annotations

from dataclasses import dataclass

from .coordinator import CoordinationResult, Evaluator, coordinate_once
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
    termination_reason: str = "step_limit"

    @property
    def states(self) -> tuple[WorldState, ...]:
        """Executed trajectory, excluding rejected proposals."""
        initial = self.steps[0].before if self.steps else self.final_state
        return (initial, *(step.result.next_state for step in self.steps if step.result.accepted))


def run(
    initial_state: WorldState, max_steps: int = 250, *, evaluator: Evaluator = evaluate
) -> SimulationResult:
    if type(max_steps) is not int or max_steps < 0:
        raise ValueError("max_steps must be a nonnegative integer")
    state = initial_state
    logs: list[StepLog] = []
    if state.goal_reached:
        return SimulationResult((), state, True, "goal_reached")
    visited = {_configuration(state)}

    for _ in range(max_steps):
        result = coordinate_once(state, evaluator=evaluator)
        logs.append(StepLog(state.timestep, state, result))

        if not result.accepted:
            # Keep the proposal in the log, but do not execute a rejected action.
            return SimulationResult(tuple(logs), state, False, "rejected")

        state = result.next_state
        if state.goal_reached:
            return SimulationResult(tuple(logs), state, True, "goal_reached")
        if all(action.kind == "WAIT" for action in result.actions.values()):
            return SimulationResult(tuple(logs), state, False, "stalled")
        configuration = _configuration(state)
        if configuration in visited:
            return SimulationResult(tuple(logs), state, False, "cycle_detected")
        visited.add(configuration)

    return SimulationResult(tuple(logs), state, False, "step_limit")


def _configuration(state: WorldState) -> tuple:
    """Identify task configurations without the clock or previous velocity."""
    return (
        tuple(
            (a.agent_id, a.position, a.carrying)
            for a in sorted(state.agents, key=lambda a: a.agent_id)
        ),
        tuple(
            (b.block_id, b.status, b.carrier, b.stack_level)
            for b in sorted(state.blocks, key=lambda b: b.block_id)
        ),
    )
