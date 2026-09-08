from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .constants import MOVE_DELTAS
from .state import Position, WorldState


@dataclass(frozen=True)
class Action:
    kind: str
    delta: Position = (0, 0)

    def __str__(self) -> str:
        return self.kind


def movement_actions() -> tuple[Action, ...]:
    return tuple(Action(name, delta) for name, delta in MOVE_DELTAS.items())


def all_actions() -> tuple[Action, ...]:
    return movement_actions() + (Action("PICKUP"), Action("DROP"))


def manhattan(a: Position, b: Position) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def adjacent(a: Position, b: Position) -> bool:
    return manhattan(a, b) == 1


def legal_actions(state: WorldState, agent_id: int, committed: Optional[dict[int, Action]] = None) -> list[Action]:
    """Return actions legal given the current state and already committed actions."""
    committed = committed or {}
    agent = state.agent(agent_id)
    actions: list[Action] = []

    for action in all_actions():
        if action.kind in MOVE_DELTAS:
            new_pos = (agent.position[0] + action.delta[0], agent.position[1] + action.delta[1])
            if not (1 <= new_pos[0] <= 7 and 1 <= new_pos[1] <= 7):
                continue
            if new_pos in state.obstacles:
                continue

            illegal = False
            for other in state.agents:
                if other.agent_id == agent_id:
                    continue
                if other.agent_id in committed:
                    committed_pos = predicted_position(other.position, committed[other.agent_id])
                    if new_pos == committed_pos:
                        illegal = True
                    if new_pos == other.position and committed[other.agent_id].kind in MOVE_DELTAS and committed_pos == agent.position:
                        illegal = True
                else:
                    # If the other agent has not yet committed, don't enter its current cell.
                    if new_pos == other.position:
                        illegal = True
            if not illegal:
                actions.append(action)

        elif action.kind == "PICKUP":
            if agent.has_block:
                continue
            if adjacent(agent.position, state.source):
                source_count = sum(b.status.value == "SOURCE" for b in state.blocks)
                committed_pickups = sum(a.kind == "PICKUP" for a in committed.values())
                if source_count - committed_pickups > 0:
                    actions.append(action)

        elif action.kind == "DROP":
            if not agent.has_block:
                continue
            if adjacent(agent.position, state.destination):
                if agent.position != state.source:
                    actions.append(action)

    return actions


def predicted_position(position: Position, action: Action) -> Position:
    if action.kind in MOVE_DELTAS:
        return (position[0] + action.delta[0], position[1] + action.delta[1])
    return position
