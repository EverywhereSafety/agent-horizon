import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

from long_horizon_rl.python_tool import PythonTool, run_python


def test_subclass_selects_runtime_and_uses_shared_worker(tmp_path):
    runtime = tmp_path / "runtime"
    (runtime / "bin").mkdir(parents=True)
    (runtime / "bin/python").touch()

    class TaskPython(PythonTool):
        def runtime_path(self):
            return runtime

    process = AsyncMock()
    process.returncode = 0
    process.communicate.return_value = (b'{"output":"42","done":false}', b"")
    with patch("asyncio.create_subprocess_exec", return_value=process) as launch:
        result = asyncio.run(TaskPython().run("print(42)"))
    assert str(runtime) in launch.call_args.args
    assert any(Path(arg).name == "python_worker.py" for arg in launch.call_args.args)
    assert result["output"] == "42"
    assert result["timeout_seconds"] == 30


def test_worker_memory_exhaustion_is_recoverable():
    process = AsyncMock()
    process.returncode = 1
    process.communicate.return_value = (b"", b"MemoryError: cannot allocate memory")
    with patch("asyncio.create_subprocess_exec", return_value=process):
        result = asyncio.run(run_python("large = []"))
    assert "MemoryError" in result["error"]
    assert result["done"] is False
