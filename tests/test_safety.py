from __future__ import annotations

from dataclasses import replace
import unittest

from src.planner.actions import Action, legal_actions
from src.planner.constraints import joint_action_legal
from src.planner.simulator import run
from src.planner.state import BlockState, BlockStatus, make_initial_state
from src.planner.transition import transition
from src.planner.visualize import ascii_world


def joint_actions(overrides=None):
    actions = {agent_id: Action("WAIT") for agent_id in (1, 2, 3)}
    actions.update(overrides or {})
    return actions


class ActionSafetyTests(unittest.TestCase):
    def test_movement_kind_supplies_its_canonical_delta(self):
        self.assertEqual(Action("RIGHT").delta, (1, 0))
        self.assertEqual(Action("WAIT").delta, (0, 0))

    def test_unknown_and_mismatched_actions_are_rejected(self):
        for kind, delta in [("TELEPORT", None), ("UP", (0, 1)), ("WAIT", (1, 0)), ("PICKUP", (1, 0))]:
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                Action(kind, delta)

    def test_transition_rejects_boundary_obstacle_and_remote_pickup(self):
        cases = [
            (make_initial_state(), Action("LEFT")),
            (make_initial_state(obstacles=frozenset({(2, 1)})), Action("RIGHT")),
            (make_initial_state(), Action("PICKUP")),
            (make_initial_state(), Action("DROP")),
        ]
        for state, action in cases:
            with self.subTest(action=action), self.assertRaises(ValueError):
                transition(state, joint_actions({1: action}))

    def test_joint_action_requires_exact_agent_ids(self):
        for actions in [{1: Action("WAIT")}, {**joint_actions(), 99: Action("WAIT")}]:
            with self.subTest(actions=actions), self.assertRaises(ValueError):
                transition(make_initial_state(), actions)

    def test_pickup_cannot_replace_an_already_carried_block(self):
        state = make_initial_state(agent_positions=((1, 6), (7, 7), (4, 4)))
        actions = joint_actions({1: Action("PICKUP")})
        carrying = transition(state, actions)
        with self.assertRaises(ValueError):
            transition(carrying, actions)
        self.assertEqual(carrying.agent(1).carrying, 1)
        self.assertEqual(carrying.block(1).carrier, 1)

    def test_simultaneous_pickups_cannot_overbook_source(self):
        state = make_initial_state(agent_positions=((1, 6), (2, 7), (4, 4)))
        state = replace(state, blocks=(BlockState(1),))
        actions = joint_actions({1: Action("PICKUP"), 2: Action("PICKUP")})
        legal, reason = joint_action_legal(state, actions)
        self.assertFalse(legal)
        self.assertIn("source blocks", reason)
        with self.assertRaises(ValueError):
            transition(state, actions)
        self.assertNotIn(Action("PICKUP"), legal_actions(state, 2, {1: Action("PICKUP")}))

    def test_following_a_departing_agent_is_legal(self):
        state = make_initial_state(agent_positions=((1, 1), (2, 1), (4, 4)))
        result = transition(state, joint_actions({1: Action("RIGHT"), 2: Action("RIGHT")}))
        self.assertEqual(result.agent(1).position, (2, 1))
        self.assertEqual(result.agent(2).position, (3, 1))

    def test_simultaneous_pickup_and_drop_ties_ignore_dict_order(self):
        state = make_initial_state(agent_positions=((1, 6), (2, 7), (4, 4)))
        actions = joint_actions({1: Action("PICKUP"), 2: Action("PICKUP")})
        carrying = transition(state, actions)
        self.assertEqual(carrying, transition(state, dict(reversed(list(actions.items())))))
        self.assertEqual([a.carrying for a in carrying.agents], [1, 2, None])
        delivery = replace(carrying, agents=(
            replace(carrying.agent(1), position=(6, 1)),
            replace(carrying.agent(2), position=(7, 2)),
            carrying.agent(3),
        ))
        drops = joint_actions({1: Action("DROP"), 2: Action("DROP")})
        stacked = transition(delivery, drops)
        self.assertEqual(stacked, transition(delivery, dict(reversed(list(drops.items())))))
        self.assertEqual(stacked.stack_height, 2)
        self.assertEqual(stacked.block(1).stack_level, 1)
        self.assertEqual(stacked.block(2).stack_level, 2)


class WorldSafetyTests(unittest.TestCase):
    def test_invalid_layouts_are_rejected(self):
        cases = [
            {"agent_positions": ((1, 1), (1, 1), (4, 4))},
            {"agent_positions": ((0, 1), (7, 7), (4, 4))},
            {"source": (8, 7)},
            {"source": (7, 1)},
            {"obstacles": frozenset({(1, 1)})},
            {"obstacles": frozenset({(1, 7)})},
            {"obstacles": frozenset({(7, 1)})},
            {"obstacles": frozenset({(8, 1)})},
        ]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                make_initial_state(**kwargs)

    def test_inconsistent_inventory_and_ids_are_rejected(self):
        state = make_initial_state()
        cases = [
            {"agents": (state.agents[0], state.agents[0])},
            {"blocks": (BlockState(1), BlockState(1))},
            {"agents": (replace(state.agent(1), carrying=1), *state.agents[1:])},
            {"blocks": (BlockState(1, BlockStatus.CARRIED, carrier=1),)},
            {"blocks": (BlockState(1, BlockStatus.SOURCE, carrier=1),)},
            {"stack_height": 1},
            {"stack_height": 1, "blocks": (BlockState(1, BlockStatus.STACKED, stack_level=2),)},
            {"timestep": -1},
        ]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                replace(state, **kwargs)

    def test_default_run_completes_and_every_transition_is_replayable(self):
        initial = make_initial_state()
        result = run(initial)
        self.assertTrue(result.success)
        self.assertEqual(result.final_state.stack_height, 4)
        self.assertEqual(initial.stack_height, 0)
        previous = initial
        for step in result.steps:
            self.assertEqual(step.before, previous)
            self.assertTrue(step.result.accepted)
            previous = transition(step.before, step.result.actions)
            self.assertEqual(previous, step.result.next_state)
            self.assertEqual(len({a.position for a in previous.agents}), len(previous.agents))
        self.assertEqual(previous, result.final_state)
        self.assertEqual(result, run(initial))

    def test_ascii_world_uses_public_renderer(self):
        rows = ascii_world(make_initial_state()).splitlines()
        self.assertEqual(len(rows), 7)
        self.assertTrue(all(len(row.split()) == 7 for row in rows))
        self.assertIn("S4", rows[-1])
        self.assertIn("D0", rows[0])


if __name__ == "__main__":
    unittest.main()
