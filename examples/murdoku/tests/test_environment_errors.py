import unittest
from unittest.mock import AsyncMock, patch
from long_horizon_rl.sandbox import SandboxError, SandboxRPCDeadlineExceeded
from murdoku_demo.murdoku import MurdokuEnvironment


class EnvironmentErrorTests(unittest.IsolatedAsyncioTestCase):
    async def test_invalid_model_argument_is_recoverable(self):
        env = MurdokuEnvironment({})
        with patch(
            "long_horizon_rl.sandbox.IsolatedEnvironment.step",
            AsyncMock(
                side_effect=SandboxError(
                    "ValueError: person must be text without line breaks"
                )
            ),
        ):
            result = await env.step(
                {"tool": "murdoku", "arguments": {"action": "place", "person": None}}
            )
        self.assertFalse(result["done"])
        self.assertIn("person must be text", result["error"])

    async def test_infrastructure_failures_remain_fatal(self):
        env = MurdokuEnvironment({})
        for failure in [
            SandboxError("worker exited"),
            SandboxRPCDeadlineExceeded("environment action state uncertain"),
        ]:
            with patch(
                "long_horizon_rl.sandbox.IsolatedEnvironment.step",
                AsyncMock(side_effect=failure),
            ):
                with self.assertRaises(type(failure)):
                    await env.step(
                        {"tool": "murdoku", "arguments": {"action": "board"}}
                    )
