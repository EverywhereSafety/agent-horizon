import asyncio
import shutil
import unittest
from pathlib import Path
from long_horizon_rl.python_workspace import PythonWorkspace
from long_horizon_rl.python_tool import run_python


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.ws = PythonWorkspace()

    def tearDown(self):
        self.ws.close()

    def test_close_exports_files_before_cleanup_and_is_idempotent(self):
        import tempfile, json

        self.ws.operate(
            {"action": "write", "path": "solver.py", "content": "print(42)"}
        )
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "workspace.json"
            self.ws.close(archive)
            self.ws.close(archive)
            self.assertIn("solver.py", json.loads(archive.read_text())["files"])
            self.assertFalse(self.ws.root.exists())

    def test_edit_snapshot_restore_and_paths(self):
        self.ws.operate(
            {"action": "write", "path": "src/solver.py", "content": "print(1)"}
        )
        self.ws.operate(
            {
                "action": "replace",
                "path": "src/solver.py",
                "old_text": "1",
                "content": "2",
            }
        )
        state = self.ws.snapshot()
        with self.assertRaises(ValueError):
            self.ws.path("../escape")
        with self.assertRaises(ValueError):
            self.ws.path("/etc/passwd")
        (self.ws.root / "link").symlink_to("/etc")
        with self.assertRaises(ValueError):
            self.ws.path("link/passwd")
        (self.ws.root / "link").unlink()
        self.ws.restore(state)
        self.assertEqual(
            self.ws.operate({"action": "read", "path": "src/solver.py"})["content"],
            "print(2)",
        )
        self.assertEqual(
            self.ws.operate({"action": "list"})["files"][0]["path"], "src/solver.py"
        )
        with self.assertRaises(ValueError):
            self.ws.operate(
                {
                    "action": "write",
                    "path": "huge",
                    "content": "x" * (self.ws.max_bytes + 1),
                }
            )


@unittest.skipUnless(shutil.which("bwrap"), "bubblewrap unavailable")
class ExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.ws = PythonWorkspace()

    async def asyncTearDown(self):
        self.ws.close()

    async def test_files_persist_imports_work_and_episodes_are_isolated(self):
        self.ws.operate({"action": "write", "path": "helper.py", "content": "value=42"})
        self.ws.operate(
            {
                "action": "write",
                "path": "solver.py",
                "content": "from helper import value\nprint(value)\nopen('answer.txt','w').write(str(value))",
            }
        )
        result = await run_python(path="solver.py", workspace=self.ws, timeout=30)
        self.assertNotIn("error", result)
        self.assertEqual(result["output"].strip(), "42")
        result = await run_python("print(open('answer.txt').read())", workspace=self.ws)
        self.assertEqual(result["output"].strip(), "42")
        other = PythonWorkspace()
        try:
            self.assertFalse((other.root / "answer.txt").exists())
        finally:
            other.close()
        with self.assertRaises(ValueError):
            await run_python("print(1)", path="solver.py", workspace=self.ws)
        with self.assertRaises(ValueError):
            await run_python("print(1)", timeout=301)

    async def test_timeout_recovery_and_hidden_host_files(self):
        result = await run_python(
            "open('checkpoint.txt','w').write('saved')\nwhile True: pass",
            workspace=self.ws,
            timeout=3,
        )
        self.assertIn("TimeoutError", result["error"])
        result = await run_python(
            "from pathlib import Path\nprint(Path('checkpoint.txt').read_text())\nprint(Path('/private/host-storage').exists())",
            workspace=self.ws,
        )
        self.assertEqual(result["output"].strip(), "saved\nFalse")

    async def test_resume_workspace_in_fresh_environment(self):
        from long_horizon_rl.sandbox import IsolatedEnvironment

        record = {
            "python_workspace": True,
            "latent_dynamics": "class LatentDynamics: pass\nclass Env:\n    def __init__(self,d,v):self.value=v",
            "initial_state_variable": 0,
            "database": {},
            "tool_schemas": [],
            "tool_implementations": [],
        }
        first = await IsolatedEnvironment(record).start()
        second = None
        try:
            await first.step(
                {
                    "tool": "workspace",
                    "arguments": {
                        "action": "write",
                        "path": "solver.py",
                        "content": "print(17)",
                    },
                }
            )
            state = await first.snapshot()
            second = await IsolatedEnvironment(record).start()
            await second.restore(state)
            result = await second.step(
                {"tool": "run_python", "arguments": {"path": "solver.py"}}
            )
            self.assertEqual(result["output"].strip(), "17")
            self.assertEqual(result["timeout_seconds"], 120)
        finally:
            await first.close()
            if second:
                await second.close()


if __name__ == "__main__":
    unittest.main()


def test_expected_file_errors_leave_workspace_usable():
    ws = PythonWorkspace()
    try:
        for action in ("read", "replace", "delete"):
            result = ws.operate(
                {
                    "action": action,
                    "path": "missing.py",
                    "old_text": "x",
                    "content": "y",
                }
            )
            assert "FileNotFoundError" in result["error"] and result["done"] is False
        (ws.root / "directory").mkdir()
        assert (
            "IsADirectoryError"
            in ws.operate({"action": "read", "path": "directory"})["error"]
        )
        ws.operate({"action": "write", "path": "solver.py", "content": "print(42)"})
        assert (
            ws.operate({"action": "read", "path": "solver.py"})["content"]
            == "print(42)"
        )
    finally:
        ws.close()


def test_workspace_infrastructure_error_is_not_feedback(monkeypatch):
    ws = PythonWorkspace()
    try:

        def broken(_):
            raise OSError("disk unavailable")

        monkeypatch.setattr(ws, "path", broken)
        import pytest

        with pytest.raises(OSError, match="disk unavailable"):
            ws.operate({"action": "read", "path": "solver.py"})
    finally:
        ws.close()


def test_directories_legacy_and_failed_restore_preserve_live_files(monkeypatch):
    import base64
    import os
    import pytest

    ws = PythonWorkspace()
    try:
        (ws.root / "empty/nested").mkdir(parents=True)
        (ws.root / "binary").write_bytes(b"\x00\xff")
        state = ws.snapshot()
        ws.restore(state)
        assert (ws.root / "empty/nested").is_dir()
        assert (ws.root / "binary").read_bytes() == b"\x00\xff"
        for invalid in ({"a": "", "a/b": ""}, {".": ""}, {"a//b": ""}, {"a": "!bad"}):
            with pytest.raises(ValueError):
                ws.restore(invalid)
            assert ws.snapshot() == state
        original_replace = os.replace

        def fail_install(source, target):
            if Path(source).name == "new":
                raise OSError("install failed")
            return original_replace(source, target)

        with monkeypatch.context() as patch:
            patch.setattr(os, "replace", fail_install)
            with pytest.raises(OSError, match="install failed"):
                ws.restore({"replacement": ""})
        assert ws.snapshot() == state
        original_write = Path.write_bytes

        def fail_write(path, data):
            if path.name == "replacement":
                raise OSError("write failed")
            return original_write(path, data)

        with monkeypatch.context() as patch:
            patch.setattr(Path, "write_bytes", fail_write)
            with pytest.raises(OSError, match="write failed"):
                ws.restore({"replacement": ""})
        assert ws.snapshot() == state
        ws.restore({"old/file": base64.b64encode(b"legacy").decode()})
        assert (ws.root / "old/file").read_bytes() == b"legacy"
        assert not (ws.root / "empty").exists()
    finally:
        ws.close()


def test_failed_rollback_retains_recovery_copy(monkeypatch):
    import os
    import pytest

    ws = PythonWorkspace()
    recovery = None
    try:
        (ws.root / "important.txt").write_text("keep")
        original = os.replace

        def fail_replacement(source, target):
            if Path(source).name in ("new", "old"):
                raise OSError("filesystem unavailable")
            return original(source, target)

        with monkeypatch.context() as patch:
            patch.setattr(os, "replace", fail_replacement)
            with pytest.raises(OSError, match="original files retained at") as error:
                ws.restore({"replacement": ""})
        recovery = Path(str(error.value).split("retained at ", 1)[1])
        assert (recovery / "important.txt").read_text() == "keep"
    finally:
        if recovery is not None:
            shutil.rmtree(recovery.parent)
        ws.close()
