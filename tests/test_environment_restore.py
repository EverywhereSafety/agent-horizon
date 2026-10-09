import shutil
import unittest
from long_horizon_rl.sandbox import IsolatedEnvironment


@unittest.skipUnless(shutil.which("bwrap"), "bubblewrap unavailable")
class RestoreTests(unittest.IsolatedAsyncioTestCase):
    def record(self):
        return {
            "latent_dynamics": """import random as rng
rng.seed(123)
progress=0
class LatentDynamics: pass
class Env:
    def __init__(self,d,value):
        self.inventory=[4,5];self.periods_done=0;self._pending_next_inv=[6,7]
""",
            "initial_state_variable": 0,
            "database": {"calls": []},
            "environment_state_globals": ["progress"],
            "tool_implementations": [
                """def advance():
    global progress
    progress+=1
    env.periods_done+=1
    env.inventory[0]+=1
    env._pending_next_inv[1]+=2
    database['calls'].append(progress)
    return {'progress':progress,'inventory':env.inventory,'pending':env._pending_next_inv,
            'periods_done':env.periods_done,'calls':database['calls'],'draw':rng.random()}
"""
            ],
            "tool_schemas": [{"name": "advance"}],
        }

    async def test_fresh_process_resume_preserves_inventory_globals_database_and_rng(
        self,
    ):
        record = self.record()
        first = await IsolatedEnvironment(record).start()
        second = None
        try:
            await first.step({"tool": "advance"})
            state = await first.snapshot()
            expected = [await first.step({"tool": "advance"}) for _ in range(4)]
            second = await IsolatedEnvironment(record).start()
            await second.restore(state)
            actual = [await second.step({"tool": "advance"}) for _ in range(4)]
            self.assertEqual(actual, expected)
        finally:
            await first.close()
            if second:
                await second.close()

    async def test_explicit_hooks_support_slot_only_environment(self):
        record = self.record()
        record["environment_snapshot_mode"] = "hooks"
        record["environment_state_globals"] = []
        record["latent_dynamics"] = """class LatentDynamics: pass
class Env:
    __slots__=('value',)
    def __init__(self,d,value):self.value=value
    def snapshot_state(self):return {'value':self.value}
    def restore_state(self,state):self.value=state['value']
"""
        record["tool_implementations"] = [
            'def advance():\n    env.value+=1\n    return {"value":env.value}'
        ]
        first = await IsolatedEnvironment(record).start()
        second = None
        try:
            await first.step({"tool": "advance"})
            state = await first.snapshot()
            expected = await first.step({"tool": "advance"})
            second = await IsolatedEnvironment(record).start()
            await second.restore(state)
            self.assertEqual(await second.step({"tool": "advance"}), expected)
        finally:
            await first.close()
            if second:
                await second.close()

    async def test_mutating_rpc_timeout_closes_worker_and_rolls_back_on_restore(self):
        from long_horizon_rl.sandbox import SandboxRPCDeadlineExceeded

        record = self.record()
        record["tool_schemas"].append({"name": "hang"})
        record["tool_implementations"].append(
            "def hang():\n    env.inventory[0]=999\n    while True: pass"
        )
        first = await IsolatedEnvironment(record).start()
        second = None
        try:
            state = await first.snapshot()
            first.timeout = 0.05
            with self.assertRaises(SandboxRPCDeadlineExceeded):
                await first.step({"tool": "hang"})
            self.assertIsNone(first.process)
            second = await IsolatedEnvironment(record).start()
            await second.restore(state)
            result = await second.step({"tool": "advance"})
            self.assertEqual(result["inventory"], [5, 5])
        finally:
            await first.close()
            if second:
                await second.close()

    async def test_context_friction_is_opt_in_and_restores(self):
        record = self.record()
        record[
            "latent_dynamics"
        ] += "\n    def apply_context_update_friction(self):\n        self.inventory[0]-=2\n        return self.inventory[0]\n"
        first = await IsolatedEnvironment(record).start()
        try:
            self.assertEqual(await first.context_update(), {"applied": False})
        finally:
            await first.close()
        record["agent_info"] = {"apply_context_update_friction": True}
        first = await IsolatedEnvironment(record).start()
        try:
            self.assertEqual(
                await first.context_update(), {"applied": True, "result": 2}
            )
            state = await first.snapshot()
            self.assertEqual((await first.context_update())["result"], 0)
            await first.restore(state)
            self.assertEqual((await first.context_update())["result"], 0)
        finally:
            await first.close()

    async def test_explicit_terminal_state_needs_no_getter(self):
        record = self.record()
        record.update(naive_baseline=0, analytical_optimal=10)
        record["tool_implementations"] = [
            'def advance():\n    return {"done":True,"final_state_variable":5}'
        ]
        first = await IsolatedEnvironment(record).start()
        try:
            self.assertEqual((await first.step({"tool": "advance"}))["reward"], 0.5)
        finally:
            await first.close()
