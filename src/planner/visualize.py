from __future__ import annotations

from .state import BlockStatus, WorldState


def ascii_world(state: WorldState) -> str:
    cells = {}
    for agent in state.agents:
        cells[agent.position] = f"A{agent.agent_id}"

    # Source and destination are shown unless occupied by an agent.
    for y in range(1, 8):
        row = []
        for x in range(1, 8):
            p = (x, y)
            if p in cells:
                token = cells[p]
            elif p in state.obstacles:
                token = "##"
            elif p == state.source:
                source_count = sum(b.status == BlockStatus.SOURCE for b in state.blocks)
                token = f"S{source_count}"
            elif p == state.destination:
                token = f"D{state.stack_height}"
            else:
                token = ".."
            row.append(f"{token:>3}")
        yield_line = " ".join(row)
        if y == 1:
            pass
        # collect via local list impossible with generator, so this helper is not a generator
    return _ascii_world_impl(state)


def _ascii_world_impl(state: WorldState) -> str:
    cells = {a.position: f"A{a.agent_id}" for a in state.agents}
    lines = []
    for y in range(1, 8):
        row = []
        for x in range(1, 8):
            p = (x, y)
            if p in cells:
                token = cells[p]
            elif p in state.obstacles:
                token = "##"
            elif p == state.source:
                token = f"S{sum(b.status == BlockStatus.SOURCE for b in state.blocks)}"
            elif p == state.destination:
                token = f"D{state.stack_height}"
            else:
                token = ".."
            row.append(f"{token:>3}")
        lines.append(" ".join(row))
    return "\n".join(lines)


def animate(result, interval: float = 0.35):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    states = [result.steps[0].before] if result.steps else [result.final_state]
    states.extend(step.result.next_state for step in result.steps)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0.5, 7.5)
    ax.set_ylim(7.5, 0.5)
    ax.set_xticks(range(1, 8))
    ax.set_yticks(range(1, 8))
    ax.grid(True)

    def draw(state: WorldState):
        ax.clear()
        ax.set_xlim(0.5, 7.5)
        ax.set_ylim(7.5, 0.5)
        ax.set_xticks(range(1, 8))
        ax.set_yticks(range(1, 8))
        ax.grid(True)
        ax.scatter([state.source[0]], [state.source[1]], marker="s", s=220, label="Source")
        ax.scatter([state.destination[0]], [state.destination[1]], marker="D", s=180, label=f"Stack {state.stack_height}")
        for obstacle in state.obstacles:
            ax.scatter([obstacle[0]], [obstacle[1]], marker="s", s=220)
        for agent in state.agents:
            ax.scatter([agent.position[0]], [agent.position[1]], s=250)
            ax.text(agent.position[0], agent.position[1], f"A{agent.agent_id}", ha="center", va="center")
        ax.set_title(f"t={state.timestep} | stacked={state.stack_height}")
        ax.legend(loc="upper right")

    def update(frame):
        draw(states[frame])
        return ax,

    animation = FuncAnimation(fig, update, frames=len(states), interval=interval * 1000, repeat=False)
    plt.show()
    return animation
