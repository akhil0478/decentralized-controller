from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

Position = Tuple[int, int]
Velocity = Tuple[int, int]


class BlockStatus(str, Enum):
    SOURCE = "SOURCE"
    CARRIED = "CARRIED"
    STACKED = "STACKED"


@dataclass(frozen=True)
class AgentState:
    agent_id: int
    position: Position
    velocity: Velocity = (0, 0)
    carrying: Optional[int] = None

    @property
    def has_block(self) -> bool:
        return self.carrying is not None


@dataclass(frozen=True)
class BlockState:
    block_id: int
    status: BlockStatus = BlockStatus.SOURCE
    carrier: Optional[int] = None
    stack_level: Optional[int] = None


@dataclass(frozen=True)
class WorldState:
    agents: Tuple[AgentState, ...]
    blocks: Tuple[BlockState, ...]
    stack_height: int = 0
    timestep: int = 0
    source: Position = (1, 7)
    destination: Position = (7, 1)
    obstacles: frozenset[Position] = field(default_factory=frozenset)

    def agent(self, agent_id: int) -> AgentState:
        for agent in self.agents:
            if agent.agent_id == agent_id:
                return agent
        raise KeyError(f"Unknown agent {agent_id}")

    def block(self, block_id: int) -> BlockState:
        for block in self.blocks:
            if block.block_id == block_id:
                return block
        raise KeyError(f"Unknown block {block_id}")

    def carried_block_id(self, agent_id: int) -> Optional[int]:
        return self.agent(agent_id).carrying

    def with_updates(
        self,
        *,
        agents: Optional[Tuple[AgentState, ...]] = None,
        blocks: Optional[Tuple[BlockState, ...]] = None,
        stack_height: Optional[int] = None,
        timestep: Optional[int] = None,
    ) -> "WorldState":
        return WorldState(
            agents=self.agents if agents is None else agents,
            blocks=self.blocks if blocks is None else blocks,
            stack_height=self.stack_height if stack_height is None else stack_height,
            timestep=self.timestep if timestep is None else timestep,
            source=self.source,
            destination=self.destination,
            obstacles=self.obstacles,
        )

    @property
    def goal_reached(self) -> bool:
        return self.stack_height == len(self.blocks) and all(not a.has_block for a in self.agents)


def make_initial_state(
    agent_positions: Tuple[Position, ...] = ((1, 1), (7, 7), (4, 4)),
    source: Position = (1, 7),
    destination: Position = (7, 1),
    obstacles: frozenset[Position] = frozenset(),
) -> WorldState:
    if len(agent_positions) != 3:
        raise ValueError("The Day 1 prototype uses exactly 3 agents.")
    agents = tuple(
        AgentState(agent_id=i + 1, position=pos)
        for i, pos in enumerate(agent_positions)
    )
    blocks = tuple(BlockState(block_id=i + 1) for i in range(4))
    return WorldState(
        agents=agents,
        blocks=blocks,
        source=source,
        destination=destination,
        obstacles=obstacles,
    )
