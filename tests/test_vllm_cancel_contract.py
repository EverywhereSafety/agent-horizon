import unittest
from types import SimpleNamespace
from long_horizon_rl.adapters.vllm_cancel_v1 import (
    abort_external_request,
    request_status,
)


class CancellationContract(unittest.IsolatedAsyncioTestCase):
    async def test_external_id_resolved_and_public_abort_used(self):
        processor = SimpleNamespace(
            external_req_ids={"external": ["internal"]},
            request_states={"internal": object()},
        )
        calls = []

        async def abort(request_id, internal):
            calls.append((request_id, internal))
            for key in processor.external_req_ids.pop(request_id):
                processor.request_states.pop(key)

        server = SimpleNamespace(
            node_rank=0, engine=SimpleNamespace(output_processor=processor, abort=abort)
        )
        self.assertTrue(request_status(server, "external")["active"])
        receipt = await abort_external_request(server, "external", False)
        self.assertTrue(receipt["aborted"])
        self.assertEqual(calls, [("external", False)])
        self.assertFalse(request_status(server, "external")["active"])
        self.assertFalse(
            (await abort_external_request(server, "external", False))["aborted"]
        )
