from __future__ import annotations

GRID_SIZE = 7
SOURCE = (1, 7)
DESTINATION = (7, 1)
BLOCK_COUNT = 4
AGENT_COUNT = 3

MOVE_DELTAS = {
    "UP": (0, -1),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
    "RIGHT": (1, 0),
    "WAIT": (0, 0),
}

ACTION_ORDER = ("UP", "DOWN", "LEFT", "RIGHT", "WAIT", "PICKUP", "DROP")
