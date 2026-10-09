"""Emit stage boundaries while retaining upstream computation and timing."""

import logging
import os
import time

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


class ProgressMixin:
    def _stage_call(self, stage, call, *args, **kwargs):
        start = time.monotonic()
        logger.info(
            "LONG_HORIZON_STAGE start step=%s stage=%s wall_time=%.6f",
            self.global_steps,
            stage,
            time.time(),
        )
        try:
            result = call(*args, **kwargs)
        except BaseException:
            logger.exception(
                "LONG_HORIZON_STAGE failed step=%s stage=%s", self.global_steps, stage
            )
            raise
        logger.info(
            "LONG_HORIZON_STAGE end step=%s stage=%s seconds=%.3f wall_time=%.6f",
            self.global_steps,
            stage,
            time.monotonic() - start,
            time.time(),
        )
        return result

    def _compute_old_log_prob(self, batch, metrics):
        return self._stage_call(
            "old_log_prob", super()._compute_old_log_prob, batch, metrics
        )

    def _update_actor(self, batch, metrics):
        rows = len(batch.keys)
        minibatch_rows = (
            self.config.actor_rollout_ref.actor.ppo_mini_batch_size
            * self.config.actor_rollout_ref.rollout.n
        )
        metrics["long_horizon/actor_rows"] = rows
        metrics["long_horizon/optimizer_minibatches_per_epoch"] = rows // minibatch_rows
        logger.info(
            "Actor batch contains %s emitted rows; %s rows per optimizer minibatch",
            rows,
            minibatch_rows,
        )
        return self._stage_call("update_actor", super()._update_actor, batch, metrics)

    def _save_checkpoint(self):
        from omegaconf import open_dict

        trainer = self.config.trainer
        minimum = trainer.get("checkpoint_min_free_bytes", 0)
        if minimum:
            from long_horizon_rl.storage_budget import checkpoint_space

            budget = checkpoint_space(trainer.default_local_dir, minimum)
            logger.info("CHECKPOINT_SPACE %s", budget)
        callback = trainer.get("checkpoint_callback_class")
        if (
            callback
            != "long_horizon_rl.adapters.checkpoint_v1.PersistentRetentionCallback"
        ):
            return self._stage_call("save_checkpoint", super()._save_checkpoint)
        # Native rank-level pruning precedes the full trainer/queue commit.
        # Delegate all deletion to our post-commit persistent callback instead.
        keep = trainer.get("max_actor_ckpt_to_keep", 2)
        prior = trainer.get("checkpoint_retention_count")
        had_count = "checkpoint_retention_count" in trainer
        with open_dict(trainer):
            trainer.checkpoint_retention_count = keep
            trainer.max_actor_ckpt_to_keep = None
        try:
            return self._stage_call("save_checkpoint", super()._save_checkpoint)
        finally:
            with open_dict(trainer):
                trainer.max_actor_ckpt_to_keep = keep
                if had_count:
                    trainer.checkpoint_retention_count = prior
                else:
                    del trainer["checkpoint_retention_count"]
