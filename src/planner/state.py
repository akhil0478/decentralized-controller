from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum

from .constants import BLOCK_COUNT, DESTINATION, GRID_SIZE, SOURCE

Position = tuple[int, int]
Velocity = tuple[int, int]


class BlockStatus(StrEnum):
    SOURCE = "SOURCE"
    CARRIED = "CARRIED"
    STACKED = "STACKED"


@dataclass(frozen=True)
class AgentState:
    agent_id: int
    position: Position
    velocity: Velocity = (0, 0)
    carrying: int | None = None

    @property
    def has_block(self) -> bool:
        return self.carrying is not None


@dataclass(frozen=True)
class BlockState:
    block_id: int
    status: BlockStatus = BlockStatus.SOURCE
    carrier: int | None = None
    stack_level: int | None = None


@dataclass(frozen=True)
class WorldState:
    agents: tuple[AgentState, ...]
    blocks: tuple[BlockState, ...]
    stack_height: int = 0
    timestep: int = 0
    source: Position = SOURCE
    destination: Position = DESTINATION
    obstacles: frozenset[Position] = field(default_factory=frozenset)
    grid_size: int = GRID_SIZE

    def __post_init__(self) -> None:
        object.__setattr__(self, "agents", tuple(self.agents))
        object.__setattr__(self, "blocks", tuple(self.blocks))
        object.__setattr__(self, "obstacles", frozenset(self.obstacles))
        if type(self.grid_size) is not int or self.grid_size < 2:
            raise ValueError("Grid size must be an integer of at least 2")
        if not self.agents:
            raise ValueError("At least one agent is required")
        if any(type(a.agent_id) is not int for a in self.agents):
            raise ValueError("Agent IDs must be integers")
        if any(type(b.block_id) is not int for b in self.blocks):
            raise ValueError("Block IDs must be integers")
        if any(type(v) is not int or v < 0 for v in (self.timestep, self.stack_height)):
            raise ValueError("Timestep and stack height must be nonnegative integers")
        if len({a.agent_id for a in self.agents}) != len(self.agents):
            raise ValueError("Agent IDs must be unique")
        if len({b.block_id for b in self.blocks}) != len(self.blocks):
            raise ValueError("Block IDs must be unique")

        positions = [a.position for a in self.agents]
        if len(set(positions)) != len(positions):
            raise ValueError("Agents cannot share a cell")
        for position in [self.source, self.destination, *positions, *self.obstacles]:
            if (
                len(position) != 2
                or any(type(v) is not int for v in position)
                or not all(1 <= v <= self.grid_size for v in position)
            ):
                raise ValueError(f"Position outside the grid: {position}")
        if self.source == self.destination:
            raise ValueError("Source and destination must be different")
        if any(p in self.obstacles for p in [self.source, self.destination, *positions]):
            raise ValueError("Obstacles cannot occupy agents, source, or destination")

        agents = {a.agent_id: a for a in self.agents}
        blocks = {b.block_id: b for b in self.blocks}
        levels = []
        for block in self.blocks:
            if block.status == BlockStatus.SOURCE:
                if block.carrier is not None or block.stack_level is not None:
                    raise ValueError("Source blocks cannot have a carrier or stack level")
            elif block.status == BlockStatus.CARRIED:
                carrier = agents.get(block.carrier)
                if (
                    carrier is None
                    or carrier.carrying != block.block_id
                    or block.stack_level is not None
                ):
                    raise ValueError("Carried block must match its carrier")
            elif block.status == BlockStatus.STACKED:
                if (
                    block.carrier is not None
                    or type(block.stack_level) is not int
                    or block.stack_level <= 0
                ):
                    raise ValueError("Stacked blocks require a stack level and no carrier")
                levels.append(block.stack_level)
            else:
                raise ValueError(f"Unknown block status: {block.status}")
        if sorted(levels) != list(range(1, self.stack_height + 1)):
            raise ValueError("Stack levels must be contiguous and match stack height")
        for agent in self.agents:
            if agent.has_block:
                block = blocks.get(agent.carrying)
                if (
                    block is None
                    or block.status != BlockStatus.CARRIED
                    or block.carrier != agent.agent_id
                ):
                    raise ValueError("Agent carrying state must match its block")

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

    def carried_block_id(self, agent_id: int) -> int | None:
        return self.agent(agent_id).carrying

    def with_updates(
        self,
        *,
        agents: tuple[AgentState, ...] | None = None,
        blocks: tuple[BlockState, ...] | None = None,
        stack_height: int | None = None,
        timestep: int | None = None,
    ) -> WorldState:
        return replace(
            self,
            agents=self.agents if agents is None else agents,
            blocks=self.blocks if blocks is None else blocks,
            stack_height=self.stack_height if stack_height is None else stack_height,
            timestep=self.timestep if timestep is None else timestep,
        )

    @property
    def goal_reached(self) -> bool:
        return self.stack_height == len(self.blocks) and all(not a.has_block for a in self.agents)


def make_initial_state(
    agent_positions: tuple[Position, ...] | None = None,
    source: Position | None = None,
    destination: Position | None = None,
    obstacles: frozenset[Position] = frozenset(),
    *,
    grid_size: int = GRID_SIZE,
    block_count: int = BLOCK_COUNT,
) -> WorldState:
    """Build a validated scenario, retaining the original demo defaults."""
    if type(grid_size) is not int or grid_size < 2:
        raise ValueError("Grid size must be an integer of at least 2")
    if type(block_count) is not int or block_count < 0:
        raise ValueError("Block count must be a nonnegative integer")
    if agent_positions is None:
        center = (grid_size + 1) // 2
        agent_positions = tuple(dict.fromkeys(((1, 1), (grid_size, grid_size), (center, center))))
    agents = tuple(
        AgentState(agent_id=i + 1, position=pos) for i, pos in enumerate(agent_positions)
    )
    blocks = tuple(BlockState(block_id=i + 1) for i in range(block_count))
    return WorldState(
        agents=agents,
        blocks=blocks,
        source=(1, grid_size) if source is None else source,
        destination=(grid_size, 1) if destination is None else destination,
        obstacles=obstacles,
        grid_size=grid_size,
    )
