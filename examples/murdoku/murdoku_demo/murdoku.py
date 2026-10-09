"""Trusted Murdoku grader/scratchpad with isolated model-written Python."""

import os
import json
from pathlib import Path
from long_horizon_rl.sandbox import IsolatedEnvironment, SandboxError


class MurdokuEnvironment(IsolatedEnvironment):
    @staticmethod
    def validate_record(record):
        if not isinstance(record.get("murdoku_case"), dict):
            raise ValueError("murdoku_case required")

    async def step(self, action):
        try:
            return await super().step(action)
        except SandboxError as exc:
            # Invalid model arguments are recoverable tool observations.
            # Process exits, RPC deadlines and unknown worker faults still fail.
            if action.get("tool") == "murdoku" and str(exc).startswith(
                ("ValueError:", "TypeError:", "KeyError:")
            ):
                return {"done": False, "error": str(exc)}
            raise

    async def finalize(self, reason):
        return await self.rpc({"op": "finalize", "reason": reason})

    def command(self):
        root = Path(os.environ["LONG_HORIZON_MURDOKU_ROOT"]).resolve()
        runtime = Path(
            os.environ.get(
                "MURDOKU_ENV_RUNTIME",
                os.environ.get("MURDOKU_TOOL_RUNTIME", str(root / ".venv")),
            )
        ).resolve()
        if (
            not (root / "murdoku_lab/environment/state.py").is_file()
            or not (runtime / "bin/python").is_file()
        ):
            raise ValueError(
                "Murdoku checkout and a working environment Python runtime are required"
            )
        worker = Path(__file__).with_name("murdoku_worker.py").resolve()
        command = [
            os.environ.get("LONG_HORIZON_BWRAP", "bwrap"),
            "--unshare-all",
            "--die-with-parent",
            "--new-session",
            "--ro-bind",
            "/usr",
            "/usr",
            "--ro-bind",
            "/lib64",
            "/lib64",
            "--ro-bind",
            "/lib",
            "/lib",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--tmpfs",
            "/tmp",
            "--dir",
            "/workspace",
            "--chdir",
            "/workspace",
            "--ro-bind",
            str(root),
            "/murdoku",
            "--ro-bind",
            str(worker),
            "/worker.py",
            "--setenv",
            "PATH",
            "/usr/bin",
            "--setenv",
            "PYTHONUNBUFFERED",
            "1",
        ]
        for bind in json.loads(os.environ.get("LONG_HORIZON_SANDBOX_BINDS", "[]")):
            path = Path(bind).resolve(strict=True)
            command += ["--ro-bind", str(path), str(path)]
        return command + [
            "--ro-bind",
            str(runtime),
            "/runtime",
            "/runtime/bin/python",
            "-I",
            "/worker.py",
        ]
