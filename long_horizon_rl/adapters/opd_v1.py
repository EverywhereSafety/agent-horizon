"""Sampled-token OPD through the pinned native trainer and loss hook."""

from functools import partial
import transfer_queue as tq
from verl.trainer.ppo.v1 import register_trainer
from verl.trainer.ppo.v1.trainer_separate_async import PPOTrainerSeparateAsync
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.tensordict_utils import get_non_tensor_data
from long_horizon_rl.episode_metrics import episode_statistics
from .opd_padding import padding_safe_opd_loss
from .progress import ProgressMixin
from .separate_lifecycle import DedicatedRolloutLifecycleMixin


@register_trainer("long_horizon_opd")
class LongHorizonOPDTrainer(
    ProgressMixin, DedicatedRolloutLifecycleMixin, PPOTrainerSeparateAsync
):
    def __init__(self, config):
        loss = config.distillation.distillation_loss
        if (
            not config.distillation.enabled
            or not loss.use_policy_gradient
            or loss.use_task_rewards
        ):
            raise ValueError(
                "long_horizon_opd requires teacher-only sampled-token policy-gradient distillation"
            )
        config.trainer.v1.trainer_mode = "separate_async"
        super().__init__(config)

    def on_init_end(self):
        super().on_init_end()
        self.actor_rollout_wg.set_loss_fn(
            partial(
                padding_safe_opd_loss,
                config=omega_conf_to_dataclass(self.config.actor_rollout_ref.actor),
                distillation_config=omega_conf_to_dataclass(self.config.distillation),
            )
        )

    def _compute_advantage(self, batch, metrics):
        # Native critic/reward tensors are zero for teacher-only OPD. Report the
        # actual task result separately, deduplicated across context segments.
        data = tq.kv_batch_get(
            keys=batch.keys,
            partition_id=batch.partition_id,
            select_fields=["extra_fields"],
        )
        extras = get_non_tensor_data(data, key="extra_fields", default=[])
        if hasattr(extras, "tolist"):
            extras = extras.tolist()
        metrics.update(episode_statistics(extras))
        return super()._compute_advantage(batch, metrics)
