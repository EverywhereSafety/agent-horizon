"""Exercise the documented query through the production loader and worker RPC."""

import io
from pathlib import Path
import subprocess
import sys

from long_horizon_rl.queries import load_queries
from long_horizon_rl.sandbox_rpc import encode_frame, read_frame


def test_documented_dynamic_query_supports_tools_and_recovery():
    root = Path(__file__).resolve().parents[1]
    records, deferred = load_queries(root / "examples/dynamic_counter.jsonl")
    assert len(records) == 1 and not deferred
    requests = [
        {"op": "init", "record": records[0]},
        {"op": "action", "action": {"tool": "advance", "arguments": {"amount": 0}}},
        {"op": "action", "action": {"tool": "advance", "arguments": {"amount": 2}}},
        {"op": "snapshot"},
        {"op": "action", "action": {"tool": "advance", "arguments": {"amount": 2}}},
        {"op": "action", "action": {"tool": "advance", "arguments": {"amount": 2}}},
    ]
    # Trusted repository fixture only; arbitrary supplied task code uses bubblewrap.
    worker = root / "long_horizon_rl/sandbox_worker.py"
    command = [sys.executable, str(worker)]
    if sys.platform == "darwin":
        # macOS does not support the Linux worker's address-space limit.
        command = [
            sys.executable,
            "-c",
            "import resource; resource.setrlimit=lambda *args:None; "
            + f'import runpy; runpy.run_path({str(worker)!r}, run_name="__main__")',
        ]
    process = subprocess.run(
        command,
        input=b"".join(encode_frame(r) for r in requests),
        capture_output=True,
        timeout=10,
        check=True,
    )
    stream = io.BytesIO(process.stdout)
    replies = list(iter(lambda: read_frame(stream), None))
    assert len(replies) == len(requests) and all(r["ok"] for r in replies)
    assert "error" in replies[1]["result"] and not replies[1]["result"]["done"]
    assert replies[-1]["result"]["reward"] == 1
    resume = [
        requests[0],
        {"op": "restore", "state": replies[3]["result"]["state"]},
        requests[4],
        requests[5],
    ]
    process = subprocess.run(
        command,
        input=b"".join(encode_frame(r) for r in resume),
        capture_output=True,
        timeout=10,
        check=True,
    )
    stream = io.BytesIO(process.stdout)
    restored = list(iter(lambda: read_frame(stream), None))
    assert restored[-1] == replies[-1]
