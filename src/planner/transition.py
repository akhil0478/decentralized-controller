from __future__ import annotations

from typing import Dict

from .actions import Action, predicted_position
from .constraints import joint_action_legal
from .state import AgentState, BlockState, BlockStatus, WorldState


def transition(state: WorldState, actions: Dict[int, Action]) -> WorldState:
    legal, reason = joint_action_legal(state, actions)
    if not legal:
        raise ValueError(f"Illegal joint transition: {reason}")

    agents = {a.agent_id: a for a in state.agents}
    blocks = {b.block_id: b for b in state.blocks}
    next_stack_height = state.stack_height

    # First apply movement.
    for agent_id, action in actions.items():
        agent = agents[agent_id]
        agents[agent_id] = AgentState(
            agent_id=agent.agent_id,
            position=predicted_position(agent.position, action),
            velocity=action.delta if action.kind in {"UP", "DOWN", "LEFT", "RIGHT"} else (0, 0),
            carrying=agent.carrying,
        )

    # Then apply pickup/drop events using the post-movement position.
    for agent_id, action in actions.items():
        agent = agents[agent_id]

        if action.kind == "PICKUP":
            source_blocks = [b for b in blocks.values() if b.status == BlockStatus.SOURCE]
            if not source_blocks:
                raise ValueError("No source block available for pickup")
            block = min(source_blocks, key=lambda b: b.block_id)
            blocks[block.block_id] = BlockState(
                block_id=block.block_id,
                status=BlockStatus.CARRIED,
                carrier=agent_id,
            )
            agents[agent_id] = AgentState(
                agent_id=agent.agent_id,
                position=agent.position,
                velocity=agent.velocity,
                carrying=block.block_id,
            )

        elif action.kind == "DROP":
            if agent.carrying is None:
                raise ValueError("Cannot drop without a carried block")
            block_id = agent.carrying
            if agent.position[0] == state.destination[0] and agent.position[1] == state.destination[1]:
                raise ValueError("Drop must occur from an adjacent cell")
            if abs(agent.position[0] - state.destination[0]) + abs(agent.position[1] - state.destination[1]) != 1:
                raise ValueError("Drop must occur from an adjacent cell")
            next_stack_height += 1
            blocks[block_id] = BlockState(
                block_id=block_id,
                status=BlockStatus.STACKED,
                carrier=None,
                stack_level=next_stack_height,
            )
            agents[agent_id] = AgentState(
                agent_id=agent.agent_id,
                position=agent.position,
                velocity=agent.velocity,
                carrying=None,
            )

    return state.with_updates(
        agents=tuple(sorted(agents.values(), key=lambda a: a.agent_id)),
        blocks=tuple(sorted(blocks.values(), key=lambda b: b.block_id)),
        stack_height=next_stack_height,
        timestep=state.timestep + 1,
    )
