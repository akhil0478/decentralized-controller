from __future__ import annotations

from .actions import Action, predicted_position
from .constraints import joint_action_legal
from .state import AgentState, BlockState, BlockStatus, WorldState


def transition(state: WorldState, actions: dict[int, Action]) -> WorldState:
    legal, reason = joint_action_legal(state, actions)
    if not legal:
        raise ValueError(f"Illegal joint transition: {reason}")

    agents = {a.agent_id: a for a in state.agents}
    blocks = {b.block_id: b for b in state.blocks}
    next_stack_height = state.stack_height

    # First apply movement.
    for agent_id, action in sorted(actions.items()):
        agent = agents[agent_id]
        agents[agent_id] = AgentState(
            agent_id=agent.agent_id,
            position=predicted_position(agent.position, action),
            velocity=action.delta if action.kind in {"UP", "DOWN", "LEFT", "RIGHT"} else (0, 0),
            carrying=agent.carrying,
        )

    # Resolve simultaneous resource ties by agent ID, independent of dict order.
    for agent_id, action in sorted(actions.items()):
        agent = agents[agent_id]

        if action.kind == "PICKUP":
            source_blocks = [b for b in blocks.values() if b.status == BlockStatus.SOURCE]
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
            block_id = agent.carrying
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
