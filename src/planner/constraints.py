from __future__ import annotations

from .actions import Action, predicted_position
from .state import WorldState


def joint_action_legal(state: WorldState, actions: dict[int, Action]) -> tuple[bool, str]:
    """Check final joint action legality. Individual actions are assumed legal already."""
    next_positions: dict[int, tuple[int, int]] = {}
    for agent in state.agents:
        if agent.agent_id not in actions:
            return False, f"Missing action for agent {agent.agent_id}"
        next_positions[agent.agent_id] = predicted_position(agent.position, actions[agent.agent_id])

    positions = list(next_positions.values())
    if len(set(positions)) != len(positions):
        return False, "Two agents enter the same cell"

    ids = [a.agent_id for a in state.agents]
    for idx, i in enumerate(ids):
        for j in ids[idx + 1 :]:
            if next_positions[i] == state.agent(j).position and next_positions[j] == state.agent(i).position:
                if next_positions[i] != state.agent(i).position or next_positions[j] != state.agent(j).position:
                    return False, "Agents swap cells in the same timestep"

    return True, "legal"
