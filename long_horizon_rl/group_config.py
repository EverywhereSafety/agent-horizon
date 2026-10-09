"""One native retain count, with an optional explicit attempt budget."""


def resolve_group_counts(
    retain, episode_config=None, *, validate=False, distillation=False
):
    settings = episode_config or {}
    if isinstance(retain, bool) or int(retain) != retain or int(retain) <= 0:
        raise ValueError("retain count must be a positive integer")
    retain = int(retain)
    asserted = settings.get("retain_n")
    if not validate and asserted is not None and asserted != retain:
        raise ValueError(
            "episode retain_n conflicts with native rollout.n / __rollout_n__"
        )
    configured = settings.get("attempt_n")
    if configured is not None and (type(configured) is not int or configured <= 0):
        raise ValueError("attempt_n must be a positive integer")
    attempts = (
        retain
        if validate or distillation
        else configured if configured is not None else retain + 2
    )
    if attempts < retain:
        raise ValueError("attempt budget is smaller than retained group")
    return attempts, retain
