import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from long_horizon_rl.environments import environment_factory, make_environment
from long_horizon_rl.queries import load_queries
from long_horizon_rl.sandbox import IsolatedEnvironment


class Entries(list):
    def select(self, group, name):
        return [e for e in self if e.name == name]


class Entry:
    name = "example_puzzle"

    def load(self):
        return Plugin


class Plugin:
    def __init__(self, record):
        self.record = record

    @staticmethod
    def validate_record(record):
        if "puzzle" not in record:
            raise ValueError("puzzle required")


class PluginTests(unittest.TestCase):
    def test_builtin_does_not_load_optional_plugins(self):
        with patch(
            "long_horizon_rl.environments.metadata.entry_points",
            side_effect=AssertionError,
        ):
            self.assertIs(environment_factory("dynamic"), IsolatedEnvironment)

    def test_external_plugin_validates_queries_and_builds_environment(self):
        record = {
            "environment_type": "example_puzzle",
            "puzzle": {},
            "messages": [],
            "tool_schemas": [{"name": "submit"}],
        }
        with patch(
            "long_horizon_rl.environments.metadata.entry_points",
            return_value=Entries([Entry()]),
        ):
            self.assertEqual(make_environment(record).record, record)
            with tempfile.TemporaryDirectory() as d:
                path = Path(d) / "queries.jsonl"
                path.write_text(json.dumps(record) + "\n")
                rows, deferred = load_queries(path)
                self.assertEqual(len(rows), 1)
                self.assertEqual(deferred, [])
                del record["puzzle"]
                path.write_text(json.dumps(record) + "\n")
                with self.assertRaisesRegex(ValueError, "puzzle required"):
                    load_queries(path)

    def test_missing_and_duplicate_plugins_fail_explicitly(self):
        for entries in [Entries(), Entries([Entry(), Entry()])]:
            with patch(
                "long_horizon_rl.environments.metadata.entry_points",
                return_value=entries,
            ):
                with self.assertRaisesRegex(ValueError, "exactly one installed plugin"):
                    environment_factory("example_puzzle")

    def test_core_has_no_task_or_platform_imports(self):
        root = Path(__file__).resolve().parents[1]
        for path in (root / "long_horizon_rl").rglob("*.py"):
            text = path.read_text()
            for forbidden in [
                "murdoku",
                "/storage/pace-apps",
                "/storage/ice1",
                "xshen332",
            ]:
                self.assertNotIn(forbidden, text, str(path))
        self.assertFalse(list((root / "scripts").glob("*.sbatch")))
