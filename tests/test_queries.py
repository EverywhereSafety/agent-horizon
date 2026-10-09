import tempfile
import json
from pathlib import Path
import unittest
from long_horizon_rl.parser import parse_action
from long_horizon_rl.queries import load_queries


class QueryTests(unittest.TestCase):
    def test_qwen_xml_and_json(self):
        expected = {"tool": "advance", "arguments": {"amount": 2}}
        self.assertEqual(
            parse_action(
                "<think>x</think><tool_call><function=advance><parameter=amount>2</parameter></function></tool_call>"
            ),
            expected,
        )
        self.assertEqual(
            parse_action(
                '<tool_call>{"name":"advance","arguments":{"amount":2}}</tool_call>'
            ),
            expected,
        )

    def test_fenced_json(self):
        text = '```json\n{"name":"wait_for_next_day","arguments":{}}\n```<|im_end|>'
        self.assertEqual(
            parse_action(text), {"tool": "wait_for_next_day", "arguments": {}}
        )

    def test_schema_required_arguments(self):
        schema = [{"name": "wait_for_next_day", "parameters": {"required": []}}]
        self.assertEqual(
            parse_action('{"name":"WaitForNextDay","arguments":{}}', schema)["tool"],
            "wait_for_next_day",
        )

    def test_external_records_are_deferred(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "q.jsonl"
            p.write_text(json.dumps({"messages": [], "env_name": "External"}) + "\n")
            supported, deferred = load_queries(p)
            self.assertEqual(supported, [])
            self.assertEqual(len(deferred), 1)


def test_nested_query_split_can_be_omitted_but_must_never_conflict():
    import pytest
    from long_horizon_rl.queries import query_from_row

    query = {"prompt_uid": "one", "split": "train", "messages": []}
    legacy = {"prompt_uid": "one", "query": query, "rollout": None}
    assert query_from_row(legacy) is query
    assert "split" not in legacy
    for changed in [{"split": "validation"}, {"split": None}, {"prompt_uid": "other"}]:
        with pytest.raises(ValueError, match="identity/split mismatch"):
            query_from_row(dict(legacy, **changed))
