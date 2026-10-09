import unittest
from unittest.mock import Mock, patch
from long_horizon_rl.trainer_plugins import load_trainer_plugin
from long_horizon_rl.async_profile import validate_async_profile


class TrainerPluginTests(unittest.TestCase):
    def test_builtin_does_not_load_external_code(self):
        with patch("long_horizon_rl.trainer_plugins.entry_points") as lookup:
            self.assertFalse(load_trainer_plugin("long_horizon_separate_async"))
            lookup.assert_not_called()

    def test_selected_installer_reports_async_contract(self):
        installer = Mock()
        installer.requires_async_checkpoint = True
        entry = Mock(name="entry")
        entry.load.return_value = installer
        points = Mock()
        points.select.return_value = [entry]
        with patch("long_horizon_rl.trainer_plugins.entry_points", return_value=points):
            self.assertTrue(load_trainer_plugin("long_horizon_external"))
        installer.assert_called_once_with()
        points.select.assert_called_once_with(
            group="long_horizon_rl.trainers", name="long_horizon_external"
        )

    def test_missing_and_duplicate_plugins_fail(self):
        for entries in [[], [Mock(), Mock()]]:
            points = Mock()
            points.select.return_value = entries
            with patch(
                "long_horizon_rl.trainer_plugins.entry_points", return_value=points
            ):
                with self.assertRaises(ValueError):
                    load_trainer_plugin("long_horizon_external")

    def test_upstream_mode_passes_through(self):
        with patch("long_horizon_rl.trainer_plugins.entry_points", return_value={}):
            self.assertFalse(load_trainer_plugin("upstream_trainer"))

    def test_external_async_mode_cannot_bypass_prefetch_validation(self):
        config = {
            "trainer": {
                "v1": {
                    "trainer_mode": "external",
                    "separate_async": {"num_warmup_batches": 1},
                    "sampler": {"max_off_policy_threshold": 1},
                }
            }
        }
        with self.assertRaises(ValueError):
            validate_async_profile(config, async_mode=True)
