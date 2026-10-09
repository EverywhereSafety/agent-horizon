"""Environment extension point; the core has no task-specific imports."""

from importlib import metadata
from ..sandbox import IsolatedEnvironment


def environment_factory(name):
    if name == "dynamic":
        return IsolatedEnvironment
    entries = metadata.entry_points()
    if hasattr(entries, "select"):
        matches = list(entries.select(group="long_horizon_rl.environments", name=name))
    else:
        matches = [
            e for e in entries.get("long_horizon_rl.environments", []) if e.name == name
        ]
    if len(matches) != 1:
        raise ValueError(
            f"environment {name!r} requires exactly one installed plugin; found {len(matches)}"
        )
    factory = matches[0].load()
    if not callable(factory):
        raise TypeError(f"environment plugin {name!r} must be callable")
    return factory


def make_environment(record):
    return environment_factory(record.get("environment_type", "dynamic"))(record)
