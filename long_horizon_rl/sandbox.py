import asyncio
import os
from pathlib import Path
from .sandbox_rpc import (
    DEFAULT_MAX_FRAME_BYTES,
    HEADER_BYTES,
    FrameError,
    validate_limit,
    encode_frame,
    frame_size,
    decode_body,
)


class SandboxError(RuntimeError):
    pass


class SandboxRPCDeadlineExceeded(SandboxError):
    pass


class SandboxRPCProtocolError(SandboxError):
    pass


class IsolatedEnvironment:
    """One networkless bubblewrap process per trajectory, with RPC deadlines.

    Dynamic code sees system runtime directories only, no home, scratch or GPU
    devices. The worker source is read-only. Real tool code never runs in the
    trainer process. Snapshots are trusted internal artifacts, never model input.
    """

    def __init__(self, record, timeout=30):
        self.frame_limit = validate_limit(
            record.get("environment_rpc_max_bytes", DEFAULT_MAX_FRAME_BYTES)
        )
        self._stderr_task = None
        self._stderr_tail = bytearray()
        self.record = record
        self.timeout = timeout
        self.process = None
        self.lock = asyncio.Lock()
        self.workspace = None
        if record.get("python_workspace"):
            from .python_workspace import PythonWorkspace

            self.workspace = PythonWorkspace()

    def command(self):
        worker = Path(__file__).with_name("sandbox_worker.py").resolve()
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
            str(worker),
            "/worker.py",
            "--ro-bind",
            str(worker.with_name("sandbox_rpc.py")),
            "/sandbox_rpc.py",
            "--setenv",
            "PATH",
            "/usr/bin",
            "--setenv",
            "PYTHONUNBUFFERED",
            "1",
            "/usr/bin/python3",
            "-I",
            "/worker.py",
            str(self.frame_limit),
        ]
        return command

    async def start(self):
        self.process = await asyncio.create_subprocess_exec(
            *self.command(),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={"PATH": "/usr/bin", "PYTHONUNBUFFERED": "1"},
        )
        self._stderr_tail.clear()
        self._stderr_task = asyncio.create_task(self._drain_stderr(self.process))
        try:
            await self.rpc({"op": "init", "record": self.record})
        except BaseException:
            await self.close()
            raise
        return self

    async def rpc(self, payload):
        async with self.lock:
            if self.process is None or self.process.returncode is not None:
                raise SandboxError("worker not running")

            async def exchange():
                self.process.stdin.write(encode_frame(payload, self.frame_limit))
                await self.process.stdin.drain()
                header = await self.process.stdout.readexactly(HEADER_BYTES)
                size = frame_size(header, self.frame_limit)
                body = await self.process.stdout.readexactly(size)
                reply = decode_body(body)
                if type(reply.get("ok")) is not bool or (
                    "result" not in reply
                    if reply["ok"]
                    else not isinstance(reply.get("error"), str)
                ):
                    raise FrameError("invalid RPC response envelope")
                return reply

            try:
                reply = await asyncio.wait_for(exchange(), self.timeout)
            except asyncio.TimeoutError as exc:
                await self.close()
                raise SandboxRPCDeadlineExceeded(
                    "environment action state uncertain; sandbox closed, no automatic retry"
                ) from exc
            except asyncio.CancelledError:
                await self.close()
                raise
            except (
                FrameError,
                asyncio.IncompleteReadError,
                BrokenPipeError,
                ConnectionResetError,
            ) as exc:
                await self.close()
                detail = self._stderr_tail.decode(errors="replace")
                raise SandboxRPCProtocolError(
                    "invalid or incomplete environment RPC; sandbox closed, no automatic retry"
                    + (": " + detail if detail else "")
                ) from exc
            if not reply["ok"]:
                raise SandboxError(reply["error"])
            return reply["result"]

    async def _drain_stderr(self, process):
        while True:
            chunk = await process.stderr.read(4096)
            if not chunk:
                break
            self._stderr_tail.extend(chunk)
            del self._stderr_tail[:-4096]

    async def step(self, action):
        if action.get("tool") == "workspace":
            if self.workspace is None:
                raise ValueError("workspace is not enabled")
            return self.workspace.operate(action.get("arguments", {}))
        if action.get("tool") == "run_python":
            from .python_tool import run_python

            args = action.get("arguments", {})
            budget = (
                self.record.get("python_timeout_seconds", 120)
                if self.workspace
                else self.timeout
            )
            return await run_python(
                args.get("code"),
                timeout=args.get("timeout", budget) if self.workspace else budget,
                workspace=self.workspace,
                path=args.get("path"),
            )
        return await self.rpc({"op": "action", "action": action})

    async def context_update(self):
        return await self.rpc({"op": "context_update"})

    async def snapshot(self):
        state = (await self.rpc({"op": "snapshot"}))["state"]
        if self.workspace is not None:
            state = {
                "_environment": state,
                "_python_workspace": self.workspace.snapshot(),
            }
        return state

    async def restore(self, state):
        files = None
        if isinstance(state, dict) and "_python_workspace" in state:
            files = state["_python_workspace"]
            state = state["_environment"]
        if files is not None and self.workspace is None:
            raise ValueError("workspace snapshot needs workspace enabled")
        result = await self.rpc({"op": "restore", "state": state})
        if self.workspace is not None:
            self.workspace.restore(files or {})
        return result

    async def close(self):
        if self.process is not None:
            process, self.process = self.process, None
            if process.returncode is None:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass

            async def discard_stdout():
                while await process.stdout.read(65536):
                    pass

            tasks = [discard_stdout(), process.wait()]
            if self._stderr_task is not None:
                tasks.append(self._stderr_task)
            await asyncio.gather(*tasks)
            self._stderr_task = None
        if self.workspace is not None:
            self.workspace.close(getattr(self, "workspace_export_path", None))
