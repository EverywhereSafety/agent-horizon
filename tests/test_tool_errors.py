import shutil
import unittest
from long_horizon_rl.sandbox import IsolatedEnvironment


@unittest.skipUnless(shutil.which("bwrap"), "bubblewrap unavailable")
class ToolErrorTests(unittest.IsolatedAsyncioTestCase):
    async def test_tool_failure_is_observed_and_environment_remains_live(self):
        record = {
            "latent_dynamics": 'class LatentDynamics: pass\nclass Env:\n def __init__(self,dynamics,initial_state_variable): self.turn=0\n def next_turn(self):\n  self.turn+=1\n  return {"done":False}\n',
            "tool_implementations": [
                'def broken():\n return {}["probed_items"]\n',
                'def inspect_state():\n return {"turn":env.turn}\n',
            ],
            "tool_schemas": [{"name": "broken"}, {"name": "inspect_state"}],
        }
        env = await IsolatedEnvironment(record).start()
        try:
            reply = await env.step({"tool": "broken", "arguments": {}})
            self.assertEqual(reply["error"], "KeyError: 'probed_items'")
            self.assertFalse(reply["done"])
            self.assertEqual(
                (await env.step({"tool": "inspect_state", "arguments": {}}))["turn"], 1
            )
            reply = await env.step({"tool": "broken", "arguments": {"unexpected": 1}})
            self.assertTrue(reply["error"].startswith("TypeError:"))
            self.assertIsNone(env.process.returncode)
        finally:
            await env.close()
