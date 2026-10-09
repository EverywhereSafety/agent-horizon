import json
import unittest
from long_horizon_rl.memory import operate
from long_horizon_rl.contracts import Config


class Encoder:
    identity = "fixture@immutable:2dim"

    def encode(self, texts):
        return [
            [1.0, 0.0] if t in ("automobile", "car", "vehicle") else [0.0, 1.0]
            for t in texts
        ]


class VectorMemoryTests(unittest.TestCase):
    def test_semantic_retrieval_update_delete_and_json_resume(self):
        m = {}
        encoder = Encoder()
        operate(m, {"operate": "archival_memory_add", "value": "apple"})
        operate(m, {"operate": "archival_memory_add", "value": "automobile"})
        args = {"operate": "archival_memory_retrieve", "query": "vehicle", "top_k": 1}
        result = operate(m, args, encoder=encoder)
        self.assertEqual(result["matches"], {"1": "automobile"})
        restored = json.loads(json.dumps(m))
        self.assertEqual(operate(restored, args, encoder=encoder), result)
        operate(
            restored, {"operate": "archival_memory_update", "id": "1", "value": "pear"}
        )
        result = operate(restored, args, encoder=encoder)
        self.assertEqual(result["scores"]["0"], 0.0)
        operate(restored, {"operate": "archival_memory_remove", "id": "1"})
        operate(restored, args, encoder=encoder)
        self.assertEqual(set(restored["_archival_vector_index"]["entries"]), {"0"})

    def test_changed_encoder_and_invalid_vectors_fail_explicitly(self):
        m = {}
        operate(m, {"operate": "archival_memory_add", "value": "car"})
        args = {"operate": "archival_memory_retrieve", "query": "vehicle"}
        operate(m, args, encoder=Encoder())
        encoder = Encoder()
        encoder.identity = "changed"
        with self.assertRaises(ValueError):
            operate(m, args, encoder=encoder)
        with self.assertRaises(ValueError):
            Config(memory_embedding_model="unversioned")
        encoder.identity = Encoder.identity
        encoder.encode = lambda x: [[float("nan"), 0.0] for _ in x]
        with self.assertRaises(ValueError):
            operate(m, args, encoder=encoder)
