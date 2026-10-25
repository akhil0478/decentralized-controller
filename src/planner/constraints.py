from __future__ import annotations

from .actions import Action, adjacent, predicted_position
from .state import BlockStatus, WorldState


def joint_action_legal(state: WorldState, actions: dict[int, Action]) -> tuple[bool, str]:
    """Validate individual actions, shared resources, and simultaneous movement."""
    ids = {a.agent_id for a in state.agents}
    if actions.keys() != ids:
        return False, "Joint action must contain exactly one action per agent"

    next_positions: dict[int, tuple[int, int]] = {}
    pickups = 0
    for agent in state.agents:
        action = actions[agent.agent_id]
        position = predicted_position(agent.position, action)
        if not all(1 <= v <= state.grid_size for v in position):
            return False, f"Agent {agent.agent_id} moves outside the grid"
        if position in state.obstacles:
            return False, f"Agent {agent.agent_id} enters an obstacle"
        if action.kind == "PICKUP":
            if agent.has_block or not adjacent(agent.position, state.source):
                return False, "Pickup requires an empty agent adjacent to the source"
            pickups += 1
        elif action.kind == "DROP":
            if not agent.has_block or not adjacent(agent.position, state.destination):
                return False, "Drop requires a carrying agent adjacent to the destination"
        next_positions[agent.agent_id] = position

    if pickups > sum(b.status == BlockStatus.SOURCE for b in state.blocks):
        return False, "Not enough source blocks for simultaneous pickups"

    positions = list(next_positions.values())
    if len(set(positions)) != len(positions):
        return False, "Two agents enter the same cell"

    ids = sorted(ids)
    for idx, i in enumerate(ids):
        for j in ids[idx + 1 :]:
            if (
                next_positions[i] == state.agent(j).position
                and next_positions[j] == state.agent(i).position
            ):
                return False, "Agents swap cells in the same timestep"

    return True, "legal"
