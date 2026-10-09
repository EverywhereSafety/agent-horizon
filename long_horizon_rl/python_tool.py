import asyncio
import json
import os
from pathlib import Path


async def _run_python(
    code=None, timeout=30, workspace=None, path=None, *, runtime, memory_bytes
):
    if (code is None) == (path is None):
        raise ValueError("provide exactly one of code or path")
    if path is not None:
        if workspace is None:
            raise ValueError("path requires an enabled workspace")
        workspace.path(path)
    elif not isinstance(code, str):
        raise ValueError("code must be a string")
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not 0 < timeout <= 300
    ):
        raise ValueError("timeout must be greater than 0 and at most 300 seconds")
    worker = Path(__file__).with_name("python_worker.py").resolve()
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
        *(
            ["--bind", str(workspace.root), "/workspace"]
            if workspace is not None
            else ["--dir", "/workspace"]
        ),
        "--chdir",
        "/workspace",
        "--ro-bind",
        str(worker),
        "/worker.py",
    ]
    runtime = Path(runtime).resolve()
    if (runtime / "bin/python").exists():
        for bind in json.loads(os.environ.get("LONG_HORIZON_SANDBOX_BINDS", "[]")):
            bind_path = Path(bind).resolve(strict=True)
            command += ["--ro-bind", str(bind_path), str(bind_path)]
        command += [
            "--ro-bind",
            str(runtime),
            "/runtime",
            "/runtime/bin/python",
            "-I",
            "/worker.py",
        ]
    else:
        command += ["/usr/bin/python3", "-I", "/worker.py"]
    started = __import__("time").monotonic()
    process = await asyncio.create_subprocess_exec(
        *command,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env={
            "PATH": "/usr/bin",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        },
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(
                json.dumps(
                    {"code": code, "path": path, "memory_bytes": memory_bytes}
                ).encode()
                + b"\n"
            ),
            timeout,
        )
        if process.returncode:
            error = stderr.decode(errors="replace")[:2000]
            if any(
                marker in error.lower()
                for marker in ("cannot allocate memory", "memoryerror", "out of memory")
            ):
                return {
                    "error": "MemoryError: isolated Python tool exhausted its memory budget",
                    "output": "",
                    "done": False,
                }
            raise RuntimeError("isolated Python tool exited: " + error)
        result = json.loads(stdout)
        result["elapsed_seconds"] = round(__import__("time").monotonic() - started, 3)
        result["timeout_seconds"] = timeout
        return result
    except asyncio.TimeoutError:
        # User-code resource exhaustion is a recoverable tool observation, not
        # an infrastructure failure or a successful terminal trajectory.
        return {
            "error": f"TimeoutError: Python execution exceeded {timeout:g} seconds",
            "output": "",
            "done": False,
            "elapsed_seconds": round(__import__("time").monotonic() - started, 3),
            "timeout_seconds": timeout,
        }
    finally:
        if process.returncode is None:
            process.kill()
        await process.wait()


class PythonTool:
    """Isolated Python execution; tasks can override runtime selection."""

    memory_bytes = 8 * 1024**3

    def runtime_path(self):
        return Path(
            os.environ.get(
                "LONG_HORIZON_TOOL_RUNTIME",
                str(Path(__file__).resolve().parent.parent / ".sandbox-env"),
            )
        )

    async def run(self, code=None, timeout=30, workspace=None, path=None):
        return await _run_python(
            code,
            timeout,
            workspace,
            path,
            runtime=self.runtime_path(),
            memory_bytes=self.memory_bytes,
        )


async def run_python(code=None, timeout=30, workspace=None, path=None):
    return await PythonTool().run(code, timeout, workspace, path)
