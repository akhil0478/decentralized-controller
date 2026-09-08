from __future__ import annotations

from dataclasses import dataclass

from .actions import manhattan
from .state import BlockStatus, WorldState


@dataclass(frozen=True)
class Score:
    """Global state score. Higher is better; tuple ordering is authoritative."""
    stack_height: int
    in_transit_blocks: int
    negative_task_distance: int
    negative_conflict_risk: int
    negative_motion_cost: int

    def as_tuple(self) -> tuple[int, int, int, int, int]:
        return (
            self.stack_height,
            self.in_transit_blocks,
            self.negative_task_distance,
            self.negative_conflict_risk,
            self.negative_motion_cost,
        )

    def scalar(self) -> float:
        return (
            self.stack_height * 100000
            + self.in_transit_blocks * 10000
            + self.negative_task_distance * 10
            + self.negative_conflict_risk * 100
            + self.negative_motion_cost
        )


def evaluate(state: WorldState, movement_cost: int = 0) -> Score:
    """Deterministic global evaluator used as the Day 1 'feel'.

    Lexicographic priorities:
      1. completed stack height
      2. blocks already in transit (pickup is meaningful progress)
      3. distance to each agent's relevant target
      4. local proximity conflict risk
      5. movement cost
    """
    in_transit = sum(b.status == BlockStatus.CARRIED for b in state.blocks)

    task_distance = 0
    for agent in state.agents:
        target = state.destination if agent.has_block else state.source
        task_distance += manhattan(agent.position, target)

    conflict_risk = 0
    positions = {agent.agent_id: agent.position for agent in state.agents}
    for i, pi in positions.items():
        for j, pj in positions.items():
            if i < j and manhattan(pi, pj) <= 1:
                conflict_risk += 1

    return Score(
        stack_height=state.stack_height,
        in_transit_blocks=in_transit,
        negative_task_distance=-task_distance,
        negative_conflict_risk=-conflict_risk,
        negative_motion_cost=-movement_cost,
    )


def is_goal(state: WorldState) -> bool:
    return state.goal_reached
