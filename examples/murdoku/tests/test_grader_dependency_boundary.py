"""The isolated trusted grader runs without importing the tool runtime package."""

import ast
import json
from pathlib import Path
import subprocess
import sys

from murdoku_lab.core.instance import Case
from murdoku_lab.core.render import cell_label


def test_native_grader_runs_without_shared_tool_imports():
    import murdoku_lab.core as core

    root = Path(core.__file__).resolve().parents[2]
    case = Case.from_json(
        json.loads((root / "tests/fixtures/quality_base_case.json").read_text())
    )
    worker = Path(__file__).resolve().parents[1] / "murdoku_demo/murdoku_worker.py"
    tree = ast.parse(worker.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and node.value == "/murdoku":
            node.value = str(root)
    source = ast.unparse(tree)
    guard = """import builtins
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == 'long_horizon_rl' or name.startswith('long_horizon_rl.'):
        raise ImportError('shared tool runtime must not be imported by grader')
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
"""
    if sys.platform == "darwin":
        # Test the RPC/import contract; Linux resource limits are qualified separately.
        guard = "import resource; resource.setrlimit=lambda *args:None\n" + guard
    requests = [
        {"op": "init", "record": {"murdoku_case": case.to_json()}},
        {
            "op": "action",
            "action": {
                "tool": "murdoku",
                "arguments": {
                    "action": "submit",
                    "placements": {
                        p: cell_label(case, k) for p, k in case.solution.items()
                    },
                    "murderer": case.answer_key,
                },
            },
        },
    ]
    process = subprocess.run(
        [sys.executable, "-I", "-c", guard + source],
        input="".join(json.dumps(r) + "\n" for r in requests),
        text=True,
        capture_output=True,
        timeout=10,
        check=True,
    )
    replies = [json.loads(line) for line in process.stdout.splitlines()]
    assert len(replies) == 2 and all(r["ok"] for r in replies)
    assert replies[-1]["result"]["reward"] == 1
    assert replies[-1]["result"]["score"]["solved"]
