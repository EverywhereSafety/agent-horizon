"""Discover optional trainer extensions without importing their implementations."""

from importlib.metadata import entry_points

BUILTIN_MODES = frozenset(
    ("separate_async", "long_horizon_separate_async", "long_horizon_opd")
)


def load_trainer_plugin(mode):
    """Register a selected extension and return its async-checkpoint requirement.

    Other upstream trainer modes pass through to veRL. Missing project-specific
    modes fail here, before workers are launched.
    """
    if mode in BUILTIN_MODES:
        return False
    points = entry_points()
    candidates = (
        points.select(group="long_horizon_rl.trainers", name=mode)
        if hasattr(points, "select")
        else [p for p in points.get("long_horizon_rl.trainers", ()) if p.name == mode]
    )
    candidates = list(candidates)
    if len(candidates) > 1:
        raise ValueError(f"Ambiguous trainer plugin: {mode}")
    if not candidates:
        if mode.startswith("long_horizon_"):
            raise ValueError(f"Trainer plugin {mode!r} is not installed")
        return False
    installer = candidates[0].load()
    if not callable(installer):
        raise TypeError(f"Trainer plugin {mode!r} must expose a callable installer")
    installer()
    return bool(getattr(installer, "requires_async_checkpoint", False))
