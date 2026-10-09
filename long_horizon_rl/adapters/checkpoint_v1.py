"""Use the upstream callback hook for persistent quota-conscious retention."""

import json
from pathlib import Path
from verl.trainer.ppo.checkpoint_callback import CheckpointCallback
from long_horizon_rl.checkpoint_retention import prune_checkpoints


class PersistentRetentionCallback(CheckpointCallback):
    def on_save(self, trainer, global_step, checkpoint_dir, async_save=False, **kwargs):
        if async_save:
            raise ValueError(
                "persistent retention requires synchronous committed checkpoints"
            )
        keep = self.config.trainer.get(
            "checkpoint_retention_count",
            self.config.trainer.get("max_actor_ckpt_to_keep", 2),
        )
        if keep is None:
            return
        removed = prune_checkpoints(Path(checkpoint_dir).parent, global_step, keep)
        receipt = Path(checkpoint_dir) / "retention.json"
        tmp = receipt.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(
                {"saved_step": global_step, "keep": keep, "removed_steps": removed}
            )
            + "\n"
        )
        tmp.replace(receipt)
        if getattr(trainer, "_last_batch_lineage_step", None) == global_step:
            receipt = Path(checkpoint_dir) / "training_lineage.json"
            tmp = receipt.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(
                    {"step": global_step, "actor_batches": trainer._last_batch_lineage},
                    ensure_ascii=False,
                )
                + "\n"
            )
            tmp.replace(receipt)
