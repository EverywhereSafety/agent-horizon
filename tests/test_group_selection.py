import json
import asyncio
import tempfile
import unittest
from long_horizon_rl.group_selection import (
    CancellationLedger,
    CancelledTrajectory,
    select_attempts,
)


class SelectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_exact_group_cancels_losers_and_persists_tombstones(self):
        with tempfile.TemporaryDirectory() as d:
            ledger = CancellationLedger(d)
            started = asyncio.Queue()
            finish = {str(i): asyncio.Event() for i in range(18)}
            closed = set()

            async def factory(uid):
                try:
                    await started.put(uid)
                    await finish[uid].wait()
                    return uid
                finally:
                    closed.add(uid)

            selection = asyncio.create_task(
                select_attempts(factory, list(finish), 16, lambda x: True, ledger)
            )
            for _ in range(18):
                await started.get()
            for i in range(16):
                finish[str(i)].set()
            result = await selection
            self.assertEqual(result, [str(i) for i in range(16)])
            self.assertEqual(closed, set(finish))
            for i in (16, 17):
                self.assertEqual(
                    json.loads(ledger.path(str(i)).read_text())[
                        "cleanup_acknowledgement"
                    ],
                    "cancelled_and_cleaned",
                )
                with self.assertRaises(CancelledTrajectory):
                    CancellationLedger(d).check(str(i))
            for uid in result:
                ledger.check(uid)

    async def test_cancel_ack_failure_prevents_publication(self):
        with tempfile.TemporaryDirectory() as d:
            started = asyncio.Event()

            async def factory(uid):
                if uid == "good":
                    await started.wait()
                    return uid
                started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    raise RuntimeError("server abort failed")

            with self.assertRaisesRegex(RuntimeError, "cancellation failed"):
                await select_attempts(
                    factory, ["good", "bad"], 1, lambda x: True, CancellationLedger(d)
                )

    async def test_shortage_settles_tasks_and_no_valid_group(self):
        with tempfile.TemporaryDirectory() as d:

            async def factory(uid):
                return uid

            with self.assertRaisesRegex(RuntimeError, "insufficient"):
                await select_attempts(
                    factory, ["a", "b"], 2, lambda x: x == "a", CancellationLedger(d)
                )
            for uid in ["a", "b"]:
                with self.assertRaises(CancelledTrajectory):
                    CancellationLedger(d).check(uid)

    async def test_external_group_cancel_cleans_every_attempt(self):
        with tempfile.TemporaryDirectory() as d:
            started = asyncio.Queue()
            closed = set()

            async def factory(uid):
                try:
                    await started.put(uid)
                    await asyncio.Event().wait()
                finally:
                    closed.add(uid)

            task = asyncio.create_task(
                select_attempts(
                    factory, ["a", "b"], 1, lambda x: True, CancellationLedger(d)
                )
            )
            await started.get()
            await started.get()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertEqual(closed, {"a", "b"})


def test_insufficient_group_surfaces_candidate_error(tmp_path):
    import asyncio, json, pytest
    from long_horizon_rl.group_selection import CancellationLedger, select_attempts

    async def fail(uid):
        raise TypeError("duplicate tools argument")

    with pytest.raises(RuntimeError, match="duplicate tools argument") as caught:
        asyncio.run(
            select_attempts(
                fail, ["a", "b"], 2, lambda value: True, CancellationLedger(tmp_path)
            )
        )
    assert isinstance(caught.value.__cause__, TypeError)
    receipt = json.loads((tmp_path / "a.failure.json").read_text())
    assert "TypeError: duplicate tools argument" in receipt["traceback"]
