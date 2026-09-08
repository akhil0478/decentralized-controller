from __future__ import annotations

import unittest

from src.planner.actions import Action, legal_actions
from src.planner.constraints import joint_action_legal
from src.planner.evaluator import evaluate
from src.planner.priority import build_priority
from src.planner.state import AgentState, make_initial_state
from src.planner.transition import transition


class CoreTests(unittest.TestCase):
    def test_initial_state(self):
        state = make_initial_state()
        self.assertEqual(len(state.agents), 3)
        self.assertEqual(len(state.blocks), 4)
        self.assertEqual(state.stack_height, 0)

    def test_priority_prefers_carrier(self):
        state = make_initial_state()
        a1 = state.agent(1)
        agents = list(state.agents)
        agents[0] = AgentState(1, a1.position, a1.velocity, carrying=1)
        state = state.with_updates(agents=tuple(agents))
        self.assertEqual(build_priority(state)[0], 1)

    def test_wall_rejects_move(self):
        state = make_initial_state(agent_positions=((1, 1), (7, 7), (4, 4)))
        actions = legal_actions(state, 1)
        self.assertNotIn("LEFT", {a.kind for a in actions})
        self.assertNotIn("UP", {a.kind for a in actions})

    def test_same_cell_rejected(self):
        state = make_initial_state(agent_positions=((1, 1), (2, 1), (4, 4)))
        legal, _ = joint_action_legal(
            state,
            {1: Action("RIGHT", (1, 0)), 2: Action("WAIT"), 3: Action("WAIT")},
        )
        self.assertFalse(legal)

    def test_swap_rejected(self):
        state = make_initial_state(agent_positions=((1, 1), (2, 1), (4, 4)))
        legal, reason = joint_action_legal(
            state,
            {1: Action("RIGHT", (1, 0)), 2: Action("LEFT", (-1, 0)), 3: Action("WAIT")},
        )
        self.assertFalse(legal)
        self.assertIn("swap", reason.lower())

    def test_pickup_adjacent(self):
        state = make_initial_state(agent_positions=((1, 6), (7, 7), (4, 4)))
        actions = legal_actions(state, 1)
        self.assertIn("PICKUP", {a.kind for a in actions})

    def test_evaluator_prefers_stack_progress(self):
        state = make_initial_state()
        better = state.with_updates(stack_height=1)
        self.assertGreater(evaluate(better).as_tuple(), evaluate(state).as_tuple())

    def test_transition_movement(self):
        state = make_initial_state()
        next_state = transition(
            state,
            {1: Action("RIGHT", (1, 0)), 2: Action("WAIT"), 3: Action("WAIT")},
        )
        self.assertEqual(next_state.agent(1).position, (2, 1))
        self.assertEqual(next_state.timestep, 1)


if __name__ == "__main__":
    unittest.main()
