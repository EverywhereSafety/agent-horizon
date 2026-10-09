import unittest
from long_horizon_rl.memory import operate


class MemoryTests(unittest.TestCase):
    def test_core_lifecycle(self):
        m = {}
        operate(m, {"operate": "core_memory_add", "key": "goal", "value": "eight"})
        self.assertEqual(
            operate(m, {"operate": "core_memory_retrieve", "key": "goal"})["value"],
            "eight",
        )
        operate(m, {"operate": "core_memory_update", "key": "goal", "value": "nine"})
        self.assertEqual(m["core"]["goal"], "nine")
        operate(m, {"operate": "core_memory_remove_all"})
        self.assertEqual(m["core"], {})

    def test_archival_lexical_retrieval(self):
        m = {}
        operate(m, {"operate": "archival_memory_add", "value": "red apple"})
        operate(m, {"operate": "archival_memory_add", "value": "blue car"})
        r = operate(
            m, {"operate": "archival_memory_retrieve", "query": "apple", "top_k": 1}
        )
        self.assertEqual(list(r["matches"].values()), ["red apple"])


class NotesTests(unittest.TestCase):
    def test_notes_are_addressable_after_state_restore_without_body_listing(self):
        import json
        from long_horizon_rl.memory import operate_notes

        m = {}
        operate_notes(m, {"action": "write", "title": "solver", "content": "print(42)"})
        restored = json.loads(json.dumps(m))
        self.assertEqual(
            operate_notes(restored, {"action": "list"}),
            {"titles": ["solver"], "done": False},
        )
        self.assertEqual(
            operate_notes(restored, {"action": "read", "title": "solver"})["content"],
            "print(42)",
        )
        operate_notes(restored, {"action": "delete", "title": "solver"})
        self.assertIn(
            "error", operate_notes(restored, {"action": "read", "title": "solver"})
        )
