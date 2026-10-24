from __future__ import annotations

import random
import unittest
from dataclasses import replace
from itertools import product

from src.planner.actions import legal_actions
from src.planner.constraints import joint_action_legal
from src.planner.coordinator import coordinate_once
from src.planner.evaluator import Score
from src.planner.simulator import run
from src.planner.state import make_initial_state
from src.planner.transition import transition
from src.planner.visualize import ascii_world


class ScenarioTests(unittest.TestCase):
    def test_custom_scenario_preserves_dimensions_and_inventory(self):
        state = make_initial_state(agent_positions=((1, 1),), grid_size=5, block_count=2)
        self.assertEqual(state.source, (1, 5))
        self.assertEqual(state.destination, (5, 1))
        result = run(state)
        self.assertTrue(result.success)
        self.assertEqual(result.termination_reason, "goal_reached")
        self.assertEqual(result.final_state.stack_height, 2)
        self.assertTrue(all(s.grid_size == 5 for s in result.states))
        self.assertEqual(len(ascii_world(result.final_state).splitlines()), 5)

    def test_larger_scenario_completes(self):
        result = run(make_initial_state(grid_size=10, block_count=6))
        self.assertTrue(result.success)
        self.assertEqual(result.final_state.stack_height, 6)

    def test_invalid_configuration_is_rejected(self):
        for kwargs in [
            {"grid_size": 1},
            {"grid_size": 3.5},
            {"block_count": -1},
            {"block_count": 1.5},
            {"agent_positions": ()},
        ]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                make_initial_state(**kwargs)

    def test_unchanged_state_decisions_do_not_depend_on_wall_clock(self):
        state = make_initial_state()
        before = coordinate_once(state)
        later = coordinate_once(replace(state, timestep=100))
        self.assertEqual(before.actions, later.actions)
        self.assertEqual(before.score, later.score)
        self.assertEqual(before.decisions, later.decisions)

    def test_seeded_joint_actions_preserve_state_and_are_order_independent(self):
        rng = random.Random(2026)
        state = make_initial_state(agent_positions=((1, 6), (2, 7), (4, 4)))
        for _ in range(100):
            candidates = [legal_actions(state, a.agent_id) for a in state.agents]
            joint_candidates = [
                dict(zip((a.agent_id for a in state.agents), actions))
                for actions in product(*candidates)
            ]
            legal = [
                actions for actions in joint_candidates if joint_action_legal(state, actions)[0]
            ]
            actions = rng.choice(legal)
            next_state = transition(state, actions)
            self.assertEqual(next_state, transition(state, dict(reversed(list(actions.items())))))
            self.assertEqual(len(next_state.blocks), len(state.blocks))
            self.assertEqual(next_state.timestep, state.timestep + 1)
            state = next_state


class TerminationTests(unittest.TestCase):
    def test_rejected_proposal_is_logged_but_never_executed(self):
        initial = make_initial_state()

        def reject_later_states(state, movement_cost):
            return Score(0, 0, 0, 0, -state.timestep)

        result = run(initial, evaluator=reject_later_states)
        self.assertFalse(result.success)
        self.assertEqual(result.termination_reason, "rejected")
        self.assertEqual(len(result.steps), 1)
        self.assertFalse(result.steps[0].result.accepted)
        self.assertEqual(result.final_state, initial)
        self.assertEqual(result.states, (initial,))

    def test_isolated_agent_reports_stall_without_exhausting_budget(self):
        state = make_initial_state(agent_positions=((1, 1),), obstacles=frozenset({(1, 2), (2, 1)}))
        result = run(state)
        self.assertEqual(result.termination_reason, "stalled")
        self.assertFalse(result.success)
        self.assertEqual(len(result.steps), 1)
        self.assertEqual(result.states[-1], result.final_state)

    def test_repeated_configuration_reports_cycle(self):
        state = make_initial_state(agent_positions=((1, 1),))

        def prefer_time(state, movement_cost):
            return Score(0, 0, 0, 0, state.timestep)

        result = run(state, evaluator=prefer_time)
        self.assertEqual(result.termination_reason, "cycle_detected")
        self.assertFalse(result.success)
        self.assertEqual(len(result.steps), 2)

    def test_zero_budget_and_already_completed_world(self):
        state = make_initial_state()
        result = run(state, max_steps=0)
        self.assertFalse(result.success)
        self.assertEqual(result.termination_reason, "step_limit")
        self.assertEqual(result.states, (state,))
        completed = run(make_initial_state(block_count=0), max_steps=0)
        self.assertTrue(completed.success)
        self.assertEqual(completed.termination_reason, "goal_reached")
        self.assertEqual(completed.steps, ())

    def test_invalid_budget_is_rejected(self):
        for budget in [-1, 0.5, True]:
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                run(make_initial_state(), max_steps=budget)


if __name__ == "__main__":
    unittest.main()
