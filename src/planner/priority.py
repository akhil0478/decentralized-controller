from __future__ import annotations

from .actions import manhattan
from .state import WorldState


def priority_key(state: WorldState, agent_id: int) -> tuple[int, int, int]:
    agent = state.agent(agent_id)
    if agent.has_block:
        return (0, manhattan(agent.position, state.destination), agent_id)
    return (1, manhattan(agent.position, state.source), agent_id)


def build_priority(state: WorldState) -> list[int]:
    return sorted((a.agent_id for a in state.agents), key=lambda agent_id: priority_key(state, agent_id))
