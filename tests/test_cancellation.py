import asyncio
import unittest
from types import SimpleNamespace
from long_horizon_rl.cancellation import cancel_aware_client


class Client:
    def __init__(self):
        self.started = asyncio.Queue()
        self.aborts = []
        self.releases = []
        self.events = {}

        async def abort(request_id, reset_prefix_cache):
            self.aborts.append((request_id, reset_prefix_cache))
            return {"aborted": True}

        self.server = SimpleNamespace(abort_request=SimpleNamespace(remote=abort))

    async def _acquire_server(self, request_id, **kwargs):
        return "server", self.server

    def _vllm_request_id(self, request_id):
        return request_id + "/actual"

    async def generate(self, request_id):
        await self._acquire_server(request_id)
        actual = self._vllm_request_id(request_id)
        try:
            event = self.events.setdefault(request_id, asyncio.Event())
            await self.started.put(actual)
            await event.wait()
            return actual
        finally:
            self.releases.append(request_id)


class CancellationTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancel_one_concurrent_request(self):
        original = Client()
        receipts = []
        wrapped = cancel_aware_client(original, on_abort=receipts.append)
        first = asyncio.create_task(wrapped.generate("first"))
        second = asyncio.create_task(wrapped.generate("second"))
        await original.started.get()
        await original.started.get()
        first.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await first
        self.assertEqual(original.aborts, [("first/actual", False)])
        self.assertEqual(
            receipts,
            [{"request_id": "first/actual", "acknowledgement": {"aborted": True}}],
        )
        original.events["second"].set()
        self.assertEqual(await second, "second/actual")
        self.assertCountEqual(original.releases, ["first", "second"])
        self.assertIs(cancel_aware_client(wrapped), wrapped)

    async def test_normal_completion_does_not_abort(self):
        original = Client()
        wrapped = cancel_aware_client(original)
        task = asyncio.create_task(wrapped.generate("normal"))
        await original.started.get()
        original.events["normal"].set()
        await task
        self.assertEqual(original.aborts, [])
