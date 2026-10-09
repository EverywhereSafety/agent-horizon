import shutil
import unittest
from long_horizon_rl.python_tool import run_python


@unittest.skipUnless(shutil.which("bwrap"), "bubblewrap unavailable")
class IsolationTests(unittest.IsolatedAsyncioTestCase):
    async def test_python_tool_output(self):
        r = await run_python("print(2+3)")
        self.assertEqual(r["output"], "5\n")

    async def test_home_and_environment_state_unavailable(self):
        r = await run_python(
            'import os; print(os.path.exists("/home/private-user")); print("env" in globals()); print(os.environ.get("HF_TOKEN"))'
        )
        self.assertEqual(r["output"], "False\nFalse\nNone\n")

    async def test_timeout_kills_worker(self):
        import asyncio

        result = await run_python("while True: pass", timeout=0.2)
        self.assertIn("TimeoutError", result["error"])
        self.assertFalse(result["done"])
        self.assertEqual((await run_python("print(7)"))["output"], "7\n")
