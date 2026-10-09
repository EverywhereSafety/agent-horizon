"""Checkpoint preflight against physical space and optional Lustre user quota."""

import errno
import os
import re
import shutil
import subprocess
from pathlib import Path


def parse_lustre_available(output):
    # lfs prints quota block counts in KiB, including an optional '*' suffix.
    for line in output.splitlines():
        match = re.match(r"^\s*(?:/\S+\s+)?(\d+)\*?\s+(\d+)\s+(\d+)\s+\S+", line)
        if match:
            used, soft, hard = map(int, match.groups())
            limits = [value for value in (soft, hard) if value > 0]
            return max(0, min(limits) - used) * 1024 if limits else None
    return None


def checkpoint_space(root, minimum_bytes):
    if type(minimum_bytes) is not int or minimum_bytes < 0:
        raise ValueError("checkpoint_min_free_bytes must be a nonnegative integer")
    path = Path(root).resolve()
    while not path.exists():
        path = path.parent
    available = shutil.disk_usage(path).free
    source = "filesystem"
    if shutil.which("lfs"):
        try:
            result = subprocess.run(
                ["lfs", "quota", "-u", str(os.getuid()), str(path)],
                capture_output=True,
                text=True,
                timeout=10,
            )
            quota = (
                parse_lustre_available(result.stdout)
                if result.returncode == 0
                else None
            )
            if quota is not None:
                available = min(available, quota)
                source = "filesystem and Lustre user quota"
        except (OSError, subprocess.TimeoutExpired):
            pass
    if available < minimum_bytes:
        raise OSError(
            errno.ENOSPC,
            "Checkpoint needs at least %d free bytes; %s reports %d. "
            "Keep the committed checkpoint and free space before retrying."
            % (minimum_bytes, source, available),
        )
    return {
        "available_bytes": available,
        "required_bytes": minimum_bytes,
        "source": source,
    }
