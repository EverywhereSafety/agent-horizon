import json
import tempfile
import unittest
from pathlib import Path

from long_horizon_rl.queries import load_queries
from murdoku_demo.teacher_dataset import (
    export_dataset,
    load_sft_examples,
    read_jsonl,
    select_sft_examples,
)


class TeacherDatasetTests(unittest.TestCase):
    def test_selection_is_distinct_reproducible_and_independent_of_input_order(self):
        rows = [{"prompt_uid": str(i)} for i in range(574)]
        chosen = select_sft_examples(rows, 560, 20261005)
        self.assertEqual(len(chosen), 560)
        self.assertEqual(
            chosen, select_sft_examples(list(reversed(rows)), 560, 20261005)
        )
        self.assertEqual(len(rows), 574)
        with self.assertRaises(ValueError):
            select_sft_examples(rows, 575)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.queries = self.root / "queries.jsonl"
        self.output = self.root / "dataset.jsonl"
        self.rows = [
            self.query("one", "train"),
            self.query("two", "train"),
            self.query("held-out", "validation"),
        ]
        self.write(self.queries, self.rows)

    def query(self, uid, split):
        return {
            "prompt_uid": uid,
            "split": split,
            "environment_type": "murdoku",
            "murdoku_case": {"solution": "PRIVATE GRADER ANSWER"},
            "assistant_turn_limit": 20,
            "agent_info": {"max_turn": 1000},
            "messages": [
                {"role": "system", "content": "Solve"},
                {"role": "user", "content": uid},
            ],
            "tool_schemas": [{"name": "murdoku", "parameters": {}}],
        }

    def write(self, path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def accepted(self, query, errors=0):
        return {
            "prompt_uid": query["prompt_uid"],
            "split": query["split"],
            "query": query,
            "rollout": {
                "prompt_uid": query["prompt_uid"],
                "replay_strict_success": True,
                "messages": query["messages"]
                + [{"role": "assistant", "content": "Submit"}],
                "tools": [],
                "turns": 1,
                "recovered_tool_errors": errors,
            },
        }

    def test_one_file_supports_rl_sft_and_held_out_queries(self):
        self.write(
            self.root / "worker-0/query_rollouts.jsonl", [self.accepted(self.rows[0])]
        )
        counts = export_dataset(self.queries, self.root, self.output)
        self.assertEqual(
            counts, {"queries": 3, "train": 2, "validation": 1, "accepted_rollouts": 1}
        )
        records, _ = load_queries(self.output, split="train")
        self.assertEqual([r["prompt_uid"] for r in records], ["one", "two"])
        self.assertEqual(records[0]["agent_info"]["max_turn"], 20)
        self.assertEqual(
            records[0]["murdoku_case"]["solution"], "PRIVATE GRADER ANSWER"
        )
        examples = load_sft_examples(self.output)
        self.assertEqual(len(examples), 1)
        self.assertNotIn("PRIVATE GRADER ANSWER", json.dumps(examples))
        self.assertEqual(load_sft_examples(self.output, split="validation"), [])
        self.assertIsNone(list(read_jsonl(self.output))[2]["rollout"])

    def test_duplicate_rollouts_choose_fewer_errors_without_duplicating_query(self):
        self.write(
            self.root / "worker-0/query_rollouts.jsonl",
            [self.accepted(self.rows[0], errors=2)],
        )
        self.write(
            self.root / "worker-1/query_rollouts.jsonl",
            [self.accepted(self.rows[0], errors=0)],
        )
        export_dataset(self.queries, self.root, self.output)
        rows = list(read_jsonl(self.output))
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["rollout_metadata"]["tool_errors"], 0)

    def test_validation_demonstrations_are_rejected(self):
        self.write(
            self.root / "worker-0/query_rollouts.jsonl", [self.accepted(self.rows[2])]
        )
        with self.assertRaises(ValueError):
            export_dataset(self.queries, self.root, self.output)

    def test_incomplete_append_tail_is_skipped_but_complete_corruption_is_not(self):
        path = self.root / "worker-0/query_rollouts.jsonl"
        self.write(path, [self.accepted(self.rows[0])])
        with path.open("a") as f:
            f.write('{"partial":')
        self.assertEqual(
            export_dataset(self.queries, self.root, self.output)["accepted_rollouts"], 1
        )
        with path.open("a") as f:
            f.write("\n")
        with self.assertRaises(json.JSONDecodeError):
            export_dataset(self.queries, self.root, self.output)

    def test_prompt_mismatch_is_rejected(self):
        row = self.accepted(self.rows[0])
        row["rollout"]["messages"] = [{"role": "system", "content": "wrong prompt"}]
        self.write(self.root / "worker-0/query_rollouts.jsonl", [row])
        with self.assertRaises(ValueError):
            export_dataset(self.queries, self.root, self.output)


def test_visual_unified_prompt_accepts_only_the_resolved_original_image(tmp_path):
    import base64
    import copy
    import pytest
    from murdoku_demo.teacher_dataset import sft_example_from_row

    image = b"\x89PNG\r\n\x1a\nfixture"
    (tmp_path / "board.png").write_bytes(image)
    query = {
        "prompt_uid": "visual-one",
        "split": "train",
        "environment_type": "murdoku",
        "murdoku_observation": "vision",
        "tool_schemas": [],
        "messages": [
            {"role": "system", "content": "original prompt"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "original clues"},
                    {"type": "image_url", "image_url": {"url": "board.png"}},
                ],
            },
        ],
    }
    messages = copy.deepcopy(query["messages"])
    messages[1]["content"][1]["image_url"]["url"] = (
        "data:image/png;base64," + base64.b64encode(image).decode()
    )
    row = {
        "prompt_uid": "visual-one",
        "query": query,
        "rollout": {
            "prompt_uid": "visual-one",
            "replay_strict_success": True,
            "messages": messages + [{"role": "assistant", "content": "submit"}],
            "tools": [],
        },
    }
    result = sft_example_from_row(row, asset_root=tmp_path)
    assert result["messages"] == row["rollout"]["messages"]
    assert query["messages"][1]["content"][1]["image_url"]["url"] == "board.png"
    for changed in ["prompt", "image"]:
        bad = copy.deepcopy(row)
        if changed == "prompt":
            bad["rollout"]["messages"][0]["content"] = "changed"
        else:
            bad["rollout"]["messages"][1]["content"][1]["image_url"][
                "url"
            ] = "data:image/png;base64,WRONG"
        with pytest.raises(ValueError, match="prompt differs"):
            sft_example_from_row(bad, asset_root=tmp_path)
    with pytest.raises(ValueError, match="prompt differs"):
        sft_example_from_row(row)
    row["query"]["split"] = "validation"
    assert sft_example_from_row(row, asset_root=tmp_path) is None


def test_mixed_dataset_routes_before_view_specific_validation(tmp_path):
    import copy
    from murdoku_demo.teacher_dataset import load_sft_examples

    text = {
        "prompt_uid": "text",
        "split": "train",
        "environment_type": "murdoku",
        "messages": [{"role": "system", "content": "original"}],
        "tool_schemas": [],
    }

    def row(query):
        return {
            "query": query,
            "rollout": {
                "prompt_uid": query["prompt_uid"],
                "replay_strict_success": True,
                "messages": copy.deepcopy(query["messages"]),
                "tools": [],
            },
        }

    vision = dict(text, prompt_uid="vision", murdoku_observation="vision")
    heldout = dict(text, prompt_uid="heldout", split="validation")
    data = [
        row(text),
        row(vision),
        row(heldout),
        {"query": dict(text, prompt_uid="empty"), "rollout": None},
    ]
    path = tmp_path / "mixed.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in data) + "\n")
    report = {}
    assert [
        x["prompt_uid"] for x in load_sft_examples(path, view="text", report=report)
    ] == ["text"]
    assert report == {"input": 4, "selected": 1, "excluded": 3}
    assert [x["prompt_uid"] for x in load_sft_examples(path, view="vision")] == [
        "vision"
    ]
    data[1]["rollout"]["replay_strict_success"] = False
    path.write_text("\n".join(json.dumps(r) for r in data) + "\n")
    import pytest

    with pytest.raises(ValueError, match="replay-qualified"):
        load_sft_examples(path, view="vision")
    assert len(load_sft_examples(path, view="text")) == 1
