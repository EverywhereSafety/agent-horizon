"""Persistent retention through verl's post-save callback (synchronous saves only)."""

import logging
import re
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def prune_checkpoints(root, saved_step, keep):
    """Prune older training-state directories only after the new save succeeded.

    Scan disk so retention also works across process restarts. Leave partial saves,
    symlinks and future steps untouched. All files within a pruned step go together.
    """
    if type(keep) is not int or keep < 1:
        raise ValueError("checkpoint retention must keep at least one complete save")
    root = Path(root)
    current = root / f"global_step_{saved_step}"
    if not (current / "data.pt").is_file() or not (current / "actor").is_dir():
        raise ValueError(
            "successful checkpoint must include actor and dataloader state"
        )
    pointer = root / "latest_checkpointed_iteration.txt"
    if not pointer.is_file() or pointer.read_text().strip() != str(saved_step):
        raise ValueError(
            "latest checkpoint pointer must commit the new step before pruning"
        )
    candidates = []
    for path in root.iterdir():
        match = re.fullmatch(r"global_step_(\d+)", path.name)
        if not match or path.is_symlink() or not path.is_dir():
            continue
        step = int(match[1])
        if (
            step <= saved_step
            and (path / "data.pt").is_file()
            and (path / "actor").is_dir()
        ):
            candidates.append((step, path))
    removed = []
    for step, path in sorted(candidates, reverse=True)[keep:]:
        shutil.rmtree(path)
        removed.append(step)
        logger.info(
            "Removed superseded checkpoint %s after committing step %s",
            path,
            saved_step,
        )
    return removed
