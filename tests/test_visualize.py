from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.planner.evaluator import Score
from src.planner.simulator import run
from src.planner.state import make_initial_state
from src.planner.visualize import animate


class AnimationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import matplotlib
        except ImportError:
            raise unittest.SkipTest(
                "Install requirements.txt to test Matplotlib rendering"
            ) from None
        matplotlib.use("Agg", force=True)

    def tearDown(self):
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_custom_grid_trajectory_renders_without_gui(self):
        import matplotlib.pyplot as plt
        from PIL import Image

        result = run(make_initial_state(agent_positions=((1, 1),), grid_size=5), max_steps=2)
        animation = animate(result, show=False)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trajectory.gif"
            animation.save(path, writer="pillow", fps=2)
            with Image.open(path) as image:
                self.assertEqual(image.n_frames, len(result.states))
        self.assertEqual(plt.gcf().axes[0].get_xlim(), (0.5, 5.5))

    def test_rejected_proposal_is_excluded_from_animation(self):
        from PIL import Image

        def reject_later_states(state, movement_cost):
            return Score(0, 0, 0, 0, -state.timestep)

        result = run(make_initial_state(), evaluator=reject_later_states)
        animation = animate(result, show=False)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rejected.gif"
            animation.save(path, writer="pillow", fps=2)
            with Image.open(path) as image:
                self.assertEqual(image.n_frames, 1)


if __name__ == "__main__":
    unittest.main()
