"""Download fixed ABI-matched public wheels and verify each release digest."""

import hashlib, json, os, platform, sys, urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

root = Path(__file__).resolve().parents[2]
pins = json.loads((root / "configs/backends/megatron_wheels.json").read_text())
if platform.machine() != "x86_64" or sys.version_info[:2] != (3, 12):
    raise RuntimeError("this qualification wheel lock targets x86_64 Python 3.12")
cache = Path(
    os.environ.get(
        "LONG_HORIZON_WHEEL_CACHE", str(root.parent / "megatron-wheel-cache")
    )
)
cache.mkdir(parents=True, exist_ok=True)


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def fetch(wheel):
    target = cache / wheel["filename"]
    if target.exists() and digest(target) == wheel["sha256"]:
        return str(target)
    tmp = target.with_suffix(".download")
    with (
        urllib.request.urlopen(wheel["url"], timeout=60) as response,
        tmp.open("wb") as f,
    ):
        while chunk := response.read(1024 * 1024):
            f.write(chunk)
        f.flush()
        os.fsync(f.fileno())
    if digest(tmp) != wheel["sha256"]:
        raise RuntimeError("wheel SHA-256 mismatch: " + wheel["name"])
    os.replace(tmp, target)
    print("VERIFIED " + wheel["name"], file=sys.stderr, flush=True)
    return str(target)


for path in ThreadPoolExecutor(4).map(fetch, pins["wheels"]):
    print(path)
