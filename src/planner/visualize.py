from __future__ import annotations

from .simulator import SimulationResult
from .state import BlockStatus, WorldState


def ascii_world(state: WorldState) -> str:
    """Render the current world, showing agents over their underlying cells."""
    cells = {a.position: f"A{a.agent_id}" for a in state.agents}
    source_count = sum(b.status == BlockStatus.SOURCE for b in state.blocks)
    lines = []
    for y in range(1, state.grid_size + 1):
        row = []
        for x in range(1, state.grid_size + 1):
            p = (x, y)
            if p in cells:
                token = cells[p]
            elif p in state.obstacles:
                token = "##"
            elif p == state.source:
                token = f"S{source_count}"
            elif p == state.destination:
                token = f"D{state.stack_height}"
            else:
                token = ".."
            row.append(f"{token:>3}")
        lines.append(" ".join(row))
    return "\n".join(lines)


def animate(result: SimulationResult, interval: float = 0.35, *, show: bool = True):
    """Animate executed states; set show=False for rendering without a GUI."""
    if interval <= 0:
        raise ValueError("Animation interval must be positive")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    states = result.states

    fig, ax = plt.subplots(figsize=(6, 6))

    def draw(state: WorldState):
        ax.clear()
        ax.set_xlim(0.5, state.grid_size + 0.5)
        ax.set_ylim(state.grid_size + 0.5, 0.5)
        ax.set_xticks(range(1, state.grid_size + 1))
        ax.set_yticks(range(1, state.grid_size + 1))
        ax.set_aspect("equal")
        ax.grid(True)
        ax.scatter([state.source[0]], [state.source[1]], marker="s", s=220, label="Source")
        ax.scatter(
            [state.destination[0]],
            [state.destination[1]],
            marker="D",
            s=180,
            label=f"Stack {state.stack_height}",
        )
        for obstacle in state.obstacles:
            ax.scatter([obstacle[0]], [obstacle[1]], marker="s", s=220)
        for agent in state.agents:
            ax.scatter([agent.position[0]], [agent.position[1]], s=250)
            ax.text(
                agent.position[0], agent.position[1], f"A{agent.agent_id}", ha="center", va="center"
            )
        ax.set_title(f"t={state.timestep} | stacked={state.stack_height}")
        ax.legend(loc="upper right")

    def update(frame):
        draw(states[frame])
        return (ax,)

    animation = FuncAnimation(
        fig, update, frames=len(states), interval=interval * 1000, repeat=False
    )
    if show:
        plt.show()
    return animation
