from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class CLITests(unittest.TestCase):
    def test_module_runs_default_scenario(self):
        result = subprocess.run(
            [sys.executable, "-m", "src.planner.main", "--json"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        summary = json.loads(result.stdout)
        self.assertTrue(summary["success"])
        self.assertEqual(summary["stack_height"], 4)
        self.assertEqual(summary["executed_steps"], summary["attempted_steps"])

    def test_script_can_run_from_outside_the_repository(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts/run_demo.py"), "--json"],
                cwd=directory,
                capture_output=True,
                text=True,
                check=True,
            )
        self.assertTrue(json.loads(result.stdout)["success"])

    def test_failed_simulation_exits_nonzero_with_summary(self):
        result = subprocess.run(
            [sys.executable, "-m", "src.planner.main", "--max-steps", "0", "--json"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 1)
        self.assertEqual(json.loads(result.stdout)["termination_reason"], "step_limit")

    def test_invalid_configuration_has_a_clear_cli_error(self):
        result = subprocess.run(
            [sys.executable, "-m", "src.planner.main", "--blocks", "-1"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("Block count must be a nonnegative integer", result.stderr)
        self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
