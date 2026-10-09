"""Native separate_async trainer with trajectory-normalized policy gradients."""

import torch
import transfer_queue as tq
from verl.trainer.ppo.v1 import register_trainer
from verl.trainer.ppo.v1.trainer_separate_async import PPOTrainerSeparateAsync
from verl.utils.tensordict_utils import (
    nested_tensor_from_tensor_list,
    get_non_tensor_data,
)
from long_horizon_rl.trajectory_weighting import trajectory_weights
from long_horizon_rl.training_lineage import training_lineage
from .progress import ProgressMixin
from .separate_lifecycle import DedicatedRolloutLifecycleMixin


@register_trainer("long_horizon_separate_async")
class TrajectoryTrainer(
    ProgressMixin, DedicatedRolloutLifecycleMixin, PPOTrainerSeparateAsync
):
    def __init__(self, config):
        actor = config.actor_rollout_ref.actor
        if (
            config.algorithm.adv_estimator != "grpo"
            or actor.loss_agg_mode != "token-sum"
        ):
            raise ValueError(
                "trajectory normalization requires GRPO and native token-sum loss"
            )
        if actor.entropy_coeff or actor.use_kl_loss:
            raise ValueError(
                "entropy/actor KL require separate trajectory normalization"
            )
        # The extension name selects this class; native runtime settings retain
        # the separate_async namespace (sync cadence, replay and recovery).
        config.trainer.v1.trainer_mode = "separate_async"
        super().__init__(config)

    def _init_resource_pool_mgr(self):
        super()._init_resource_pool_mgr()
        if self.config.trainer.get("model_aware_microbatching", False):
            if self.config.trainer.get(
                "policy_reduction", "trajectory"
            ) == "legacy_row_microbatch" and not self.config.trainer.get(
                "allow_partition_dependent_reduction", False
            ):
                raise ValueError(
                    "model-aware batching changes legacy reduction; explicitly acknowledge the partition-dependent objective"
                )
            import ray
            from verl.trainer.ppo.v1.trainer_base import Role
            from long_horizon_rl.diagnostics.model_cost_v1 import (
                ModelAwareActorRolloutRefWorker,
            )

            for role in (Role.ActorRollout, Role.ActorRolloutRef):
                if role in self.role_worker_mapping:
                    self.role_worker_mapping[role] = ray.remote(
                        ModelAwareActorRolloutRefWorker
                    )

    def fit(self, agent_loop_manager):
        if self.config.trainer.get("live_abort_probe", False):
            import asyncio
            from long_horizon_rl.diagnostics.live_abort_probe import probe

            asyncio.run(probe(self))
            return
        return super().fit(agent_loop_manager)

    def _fetch_one_gen_batch(self):
        if not self.config.trainer.get("single_pass_unique_queries", False):
            return super()._fetch_one_gen_batch()
        if self.config.trainer.resume_mode != "disable":
            raise ValueError("single-pass pilot requires resume disabled")
        if not hasattr(self, "_single_pass_guard"):
            from long_horizon_rl.single_pass import SinglePassGuard

            limit = len(self.train_dataloader.dataset)
            if limit != self.config.trainer.get("unique_query_limit", limit):
                raise ValueError(
                    "filtered training dataset differs from requested unique-query count"
                )
            self._single_pass_guard = SinglePassGuard(limit)
        size = (
            self.config.data.get("gen_batch_size", None)
            or self.config.data.train_batch_size
        )
        self._single_pass_guard.before_fetch(size)
        batch = super()._fetch_one_gen_batch()
        records = get_non_tensor_data(batch, key="record_json", default=None)
        if hasattr(records, "tolist"):
            records = records.tolist()
        if records is None:
            raise ValueError("single-pass experiment requires logical record IDs")
        import json

        self._single_pass_guard.admit([json.loads(r)["prompt_uid"] for r in records])
        return batch

    def _add_batch_to_generate(self):
        if self.config.trainer.get("resume_drain_only", False):
            if not self._restored_tq_prompt_count:
                raise RuntimeError(
                    "drain-only recovery requires checkpointed prompt groups"
                )
            return
        return super()._add_batch_to_generate()

    def _add_prompts_to_generate(self, num_prompts):
        if self.config.trainer.get("resume_drain_only", False):
            raise RuntimeError(
                "drain-only recovery cannot refill from the dataset; inspect evicted/failed groups"
            )
        return super()._add_prompts_to_generate(num_prompts)

    def _compute_advantage(self, batch, metrics):
        batch = super()._compute_advantage(batch, metrics)
        data = tq.kv_batch_get(
            keys=batch.keys,
            partition_id=batch.partition_id,
            select_fields=[
                "advantages",
                "response_mask",
                "record_json",
                "extra_fields",
            ],
        )
        counts = [int(row.sum().item()) for row in data["response_mask"].unbind()]
        records = get_non_tensor_data(data, key="record_json", default=None)
        if hasattr(records, "tolist"):
            records = records.tolist()
        if isinstance(records, str):
            records = [records] * len(batch.keys)
        if getattr(self, "_last_batch_lineage_step", None) != self.global_steps:
            self._last_batch_lineage_step = self.global_steps
            self._last_batch_lineage = []
            self._last_batch_episode_extras = []
        self._last_batch_lineage.append(training_lineage(batch.keys, counts, records))
        extras = get_non_tensor_data(data, key="extra_fields", default=[])
        if hasattr(extras, "tolist"):
            extras = extras.tolist()
        for key, extra in zip(batch.keys, extras):
            if isinstance(extra, dict):
                self._last_batch_episode_extras.append(
                    {**extra, "native_prompt_uid": key.rsplit("_", 2)[0]}
                )
        from long_horizon_rl.episode_metrics import episode_statistics

        metrics.update(episode_statistics(self._last_batch_episode_extras))
        mini_size = (
            self.config.actor_rollout_ref.actor.ppo_mini_batch_size
            * self.config.actor_rollout_ref.rollout.n
        )
        if len(batch.keys) % mini_size:
            raise ValueError("batch must be aligned to complete native minibatches")
        weights = trajectory_weights(batch.keys, counts, len(batch.keys) // mini_size)
        objective_fields = self._objective_fields(data, weights)
        values = [
            row * weight for row, weight in zip(data["advantages"].unbind(), weights)
        ]
        data["advantages"] = nested_tensor_from_tensor_list(values, ragged_idx=1)
        tq.kv_batch_put(
            keys=batch.keys,
            partition_id=batch.partition_id,
            fields=data.select("advantages", *objective_fields),
        )
        if batch.fields is not None:
            batch.fields = list(dict.fromkeys([*batch.fields, *objective_fields]))
        metrics["long_horizon/trajectory_weight_min"] = min(weights)
        metrics["long_horizon/trajectory_weight_max"] = max(weights)
        return batch

    def _objective_fields(self, data, weights):
        return []
