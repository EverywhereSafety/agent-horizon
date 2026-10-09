"""Reject statically unknown agent names before allocating model workers."""


def validate_dataset_agents(dataset, allowed, default):
    if default not in allowed:
        raise ValueError(f"default agent loop {default!r} is not registered/configured")
    frame = getattr(dataset, "dataframe", None)
    if frame is None or not hasattr(frame, "column_names"):
        return False  # Custom/streaming datasets retain their runtime contract.
    if "agent_name" not in frame.column_names:
        return True
    if not hasattr(frame, "unique"):
        return False
    names = frame.unique("agent_name")
    unknown = [
        name for name in names if not isinstance(name, str) or name not in allowed
    ]
    if unknown:
        raise ValueError(
            f"dataset contains unregistered agent loops: {unknown!r}; "
            f"configured names: {sorted(allowed)!r}"
        )
    return True
