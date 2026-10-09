"""Exclude synthetic all-masked microbatches from OPD empty reductions."""


def padding_safe_opd_loss(
    model_output, data, dp_group=None, *, config, distillation_config
):
    mask = data["response_mask"]
    values = mask.values() if mask.is_nested else mask
    if not values.bool().any():
        log_probs = model_output["log_probs"]
        log_probs = log_probs.values() if log_probs.is_nested else log_probs
        return log_probs.sum() * 0.0, {}
    from verl.trainer.distillation.losses import distillation_ppo_loss

    return distillation_ppo_loss(
        config=config,
        distillation_config=distillation_config,
        model_output=model_output,
        data=data,
        dp_group=dp_group,
    )
