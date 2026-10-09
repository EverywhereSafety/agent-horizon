import tempfile
import unittest
from pathlib import Path
from long_horizon_rl.checkpoint_retention import prune_checkpoints


class RetentionTests(unittest.TestCase):
    def test_restart_scan_preserves_latest_two_and_partial_saves(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for step in (1, 2, 3):
                path = root / f"global_step_{step}"
                (path / "actor").mkdir(parents=True)
                (path / "data.pt").write_text("state")
                (path / "transfer_queue").mkdir()
            (root / "global_step_4").mkdir()
            (root / "latest_checkpointed_iteration.txt").write_text("3")
            self.assertEqual(prune_checkpoints(root, 3, 2), [1])
            self.assertFalse((root / "global_step_1").exists())
            for step in (2, 3, 4):
                self.assertTrue((root / f"global_step_{step}").exists())
            self.assertEqual(prune_checkpoints(root, 3, 2), [])

    def test_uncommitted_save_never_removes_previous(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for step in (1, 2):
                path = root / f"global_step_{step}"
                (path / "actor").mkdir(parents=True)
                (path / "data.pt").write_text("state")
            (root / "latest_checkpointed_iteration.txt").write_text("1")
            with self.assertRaises(ValueError):
                prune_checkpoints(root, 2, 1)
            self.assertTrue((root / "global_step_1").exists())
