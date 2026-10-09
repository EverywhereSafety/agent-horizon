import unittest
from types import SimpleNamespace
from omegaconf import OmegaConf
from long_horizon_rl.adapters.progress import ProgressMixin


class NativeDouble:
    def _compute_old_log_prob(self, batch, metrics):
        return batch

    def _update_actor(self, batch, metrics):
        return batch

    def _save_checkpoint(self):
        return "saved"


class Trainer(ProgressMixin, NativeDouble):
    global_steps = 7
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "actor": {"ppo_mini_batch_size": 1},
                "rollout": {"n": 2},
            },
            "trainer": {},
        }
    )


class ProgressTests(unittest.TestCase):
    def test_diagnostics_preserve_results_and_report_actual_optimizer_count(self):
        trainer = Trainer()
        batch = SimpleNamespace(keys=list(range(32)))
        metrics = {}
        with self.assertLogs("long_horizon_rl.adapters.progress", level="INFO") as log:
            self.assertIs(trainer._compute_old_log_prob(batch, metrics), batch)
            self.assertIs(trainer._update_actor(batch, metrics), batch)
            self.assertEqual(trainer._save_checkpoint(), "saved")
        self.assertEqual(metrics["long_horizon/optimizer_minibatches_per_epoch"], 16)
        self.assertEqual(
            sum("LONG_HORIZON_STAGE start" in item for item in log.output), 3
        )
        self.assertEqual(
            sum("LONG_HORIZON_STAGE end" in item for item in log.output), 3
        )

    def test_failed_stage_is_logged_and_original_error_propagates(self):
        trainer = Trainer()
        error = RuntimeError("native failure")

        def fail():
            raise error

        with self.assertLogs("long_horizon_rl.adapters.progress", level="ERROR") as log:
            with self.assertRaises(RuntimeError) as raised:
                trainer._stage_call("probe", fail)
        self.assertIs(raised.exception, error)
        self.assertIn("stage=probe", log.output[0])


class RetentionNativeDouble:
    def _save_checkpoint(self):
        assert self.config.trainer.max_actor_ckpt_to_keep is None
        assert self.config.trainer.checkpoint_retention_count == 1
        if self.fail:
            raise RuntimeError("queue save failed")
        return "committed"


class RetentionTrainer(ProgressMixin, RetentionNativeDouble):
    global_steps = 2
    fail = False

    def __init__(self):
        self.config = OmegaConf.create(
            {
                "trainer": {
                    "max_actor_ckpt_to_keep": 1,
                    "checkpoint_callback_class": "long_horizon_rl.adapters.checkpoint_v1.PersistentRetentionCallback",
                }
            }
        )
        OmegaConf.set_struct(self.config, True)


class RetentionBoundaryTests(unittest.TestCase):
    def test_worker_pruning_disabled_until_full_commit(self):
        trainer = RetentionTrainer()
        self.assertEqual(trainer._save_checkpoint(), "committed")
        self.assertEqual(trainer.config.trainer.max_actor_ckpt_to_keep, 1)
        self.assertNotIn("checkpoint_retention_count", trainer.config.trainer)

    def test_failed_save_restores_config_without_rank_pruning(self):
        trainer = RetentionTrainer()
        trainer.fail = True
        with self.assertLogs("long_horizon_rl.adapters.progress", level="ERROR"):
            with self.assertRaisesRegex(RuntimeError, "queue save failed"):
                trainer._save_checkpoint()
        self.assertEqual(trainer.config.trainer.max_actor_ckpt_to_keep, 1)
        self.assertNotIn("checkpoint_retention_count", trainer.config.trainer)


from long_horizon_rl.diagnostics.overlap_report import stage_intervals, summarize


def test_failed_and_unfinished_stages_are_not_reported_as_complete():
    log = """LONG_HORIZON_STAGE start step=1 stage=update_actor wall_time=10.0
LONG_HORIZON_STAGE failed step=1 stage=update_actor
LONG_HORIZON_STAGE start step=2 stage=old_log_prob wall_time=20.0
LONG_HORIZON_STAGE end step=2 stage=old_log_prob seconds=3.0 wall_time=23.0
"""
    assert stage_intervals(log) == [(2, "old_log_prob", 20, 23)]


def test_overlap_requires_simultaneous_activity_and_complete_device_samples():
    intervals = [(1, "update_actor", 10, 13)]
    samples = {
        9: {0: 100, 1: 100},
        10: {0: 100, 1: 0},
        11: {0: 100, 1: 80},
        12: {0: 0, 1: 100},
        13: {0: 100},
    }
    (report,) = summarize(intervals, samples, 0, 2, 20)
    assert report["samples"] == 3
    assert report["trainer_active_samples"] == 2
    assert report["overlap_samples"] == 1
    assert report["overlap_fraction_of_trainer_active"] == 0.5


from long_horizon_rl.episode_metrics import episode_statistics


def test_segments_do_not_bias_episode_or_group_metrics():
    a = {
        "trajectory_uid": "a",
        "native_prompt_uid": "group",
        "reward_extra_info": {
            "reward": 0,
            "submitted": 0,
            "strict_success": False,
            "turns": 10,
        },
    }
    b = {
        "trajectory_uid": "b",
        "native_prompt_uid": "group",
        "reward_extra_info": {
            "reward": 1,
            "submitted": 1,
            "strict_success": True,
            "turns": 2,
        },
    }
    result = episode_statistics([a, a, a, b])
    assert result["episodes/count"] == 2
    assert result["episodes/reward_mean"] == 0.5
    assert result["episodes/strict_success_mean"] == 0.5
    assert result["episodes/turns_mean"] == 6
    assert result["episodes/groups_with_reward_variation"] == 1


def test_unreported_success_is_not_invented_as_zero():
    result = episode_statistics(
        [
            {
                "trajectory_uid": "a",
                "reward_extra_info": {"reward": 0, "strict_success": None},
            }
        ]
    )
    assert "episodes/strict_success_mean" not in result
    assert result["episodes/nonzero_reward_fraction"] == 0
