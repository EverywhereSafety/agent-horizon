import unittest
from long_horizon_rl.transport import TokenHTTPBackend


class TransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_exact_ids_and_request_lineage(self):
        captured = []

        async def request(payload):
            captured.append(payload)
            return {
                "output_token_ids": [123, 456],
                "output_logprobs": [-0.1, -0.2],
                "text": "different decoded representation",
                "served_weight_version": 3,
            }

        backend = TokenHTTPBackend("http://fixture", request=request)
        generation = await backend.generate([8, 9], 2, 2, "episode", 7)
        self.assertEqual(generation.token_ids, [123, 456])
        self.assertEqual(captured[0]["input_ids"], [8, 9])
        self.assertEqual(captured[0]["request_id"], "episode/7")

    async def test_missing_serving_version_fails_closed(self):
        async def request(_):
            return {"output_token_ids": [1], "output_logprobs": [-0.1], "text": "x"}

        backend = TokenHTTPBackend("http://fixture", request=request)
        with self.assertRaises(KeyError):
            await backend.generate([], 1, 0, "e", 0)

    async def test_stale_server_rejected(self):
        async def request(_):
            return {
                "output_token_ids": [1],
                "output_logprobs": [-0.1],
                "text": "x",
                "served_weight_version": 0,
            }

        backend = TokenHTTPBackend("http://fixture", request=request)
        with self.assertRaises(ValueError):
            await backend.generate([], 1, 1, "e", 0)
