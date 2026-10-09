"""Validate native async prefetch settings without importing the GPU runtime."""


def validate_async_profile(config, *, async_mode=False):
    trainer = config["trainer"]
    if not async_mode and trainer["v1"]["trainer_mode"] not in (
        "separate_async",
        "long_horizon_separate_async",
        "long_horizon_opd",
    ):
        return
    settings = trainer["v1"]["separate_async"]
    warmup = settings.get("num_warmup_batches", 0)
    if not isinstance(warmup, int) or isinstance(warmup, bool) or warmup < 0:
        raise ValueError("num_warmup_batches must be a nonnegative integer")
    span = trainer["v1"]["sampler"]["max_off_policy_threshold"]
    if not isinstance(span, int) or isinstance(span, bool) or span < 1:
        raise ValueError(
            "max_off_policy_threshold is an inclusive positive version span"
        )
    if warmup and span <= warmup:
        raise ValueError(
            "async prefetch needs a version span greater than its warmup batch count; "
            "otherwise prefetched groups expire before use"
        )
    if warmup and trainer.get("single_pass_unique_queries", False):
        raise ValueError(
            "single-pass experiments cannot prefetch past their fixed dataset boundary; "
            "set num_warmup_batches=0"
        )
