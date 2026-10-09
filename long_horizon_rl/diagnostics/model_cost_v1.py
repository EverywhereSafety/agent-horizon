"""Scoped native partition adapter; no vendor source changes or new DP collectives."""

from functools import wraps
from long_horizon_rl.diagnostics.model_cost import (
    ModelCost,
    _active_cost,
    rebalance_partitions,
    use_model_cost,
)


def install_partition_adapter():
    from verl.utils import seqlen_balancing as sb
    from verl.workers.engine import utils

    if getattr(sb.rearrange_micro_batches, "_long_horizon_cost", False):
        return
    native = sb.rearrange_micro_batches
    workload = sb.calculate_workload

    def calculate(lengths):
        cost = _active_cost.get()
        return workload(lengths) if cost is None else cost.workload(lengths)

    @wraps(native)
    def partition(*args, **kwargs):
        micro, indices = native(*args, **kwargs)
        cost = _active_cost.get()
        if cost is None:
            return micro, indices
        import inspect
        from verl.utils import tensordict_utils as tu

        params = inspect.signature(native).bind(*args, **kwargs)
        params.apply_defaults()
        data = params.arguments["batch"]
        capacity = params.arguments["max_token_len"]
        force = params.arguments["force_group_size"]
        ids = data["input_ids"]
        lengths = (
            (ids.offsets().diff() if ids.is_nested else data["attention_mask"].sum(-1))
            .cpu()
            .tolist()
        )
        if force > 1:
            # A group cost must sum individual squared lengths; cannot replace
            # that by squared group length. Keep native model-aware grouping.
            return micro, indices
        indices = rebalance_partitions(lengths, indices, capacity, cost)
        indices.sort(
            key=lambda b: sum(
                cost.quadratic * lengths[i] ** 2 + cost.linear * lengths[i] for i in b
            ),
            reverse=True,
        )
        indices = indices[::2][::-1] + indices[1::2]
        return [tu.index_select_tensor_dict(data, idx) for idx in indices], indices

    partition._long_horizon_cost = True
    sb.calculate_workload = calculate
    sb.rearrange_micro_batches = partition
    utils.rearrange_micro_batches = partition


def attach_model_cost(engine, cost):
    install_partition_adapter()
    native = engine.forward_backward_batch

    @wraps(native)
    def forward(*args, **kwargs):
        with use_model_cost(cost):
            return native(*args, **kwargs)

    engine.forward_backward_batch = forward


# Native actor construction remains responsible for training/serving engines.
from verl.experimental.separation.engine_workers import DetachActorWorker
from verl.single_controller.base.decorator import register, Dispatch


class ModelAwareActorRolloutRefWorker(DetachActorWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        result = super().init_model()
        if "actor" in self.role:
            settings = self.config.actor.get("model_cost", {})
            if settings:
                cost = ModelCost(
                    float(settings["quadratic"]), float(settings["linear"])
                )
            else:
                cost = ModelCost.from_config(self.actor.model_config.hf_config)
            attach_model_cost(self.actor.engine, cost)
        return result
