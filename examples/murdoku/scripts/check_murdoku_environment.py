"""Exercise real isolated scratchpad, hidden-key scoring, restore and Python isolation."""

import argparse, asyncio, json, os
from pathlib import Path
from long_horizon_rl.environments import make_environment


async def main():
    root = Path(os.environ["LONG_HORIZON_MURDOKU_ROOT"])
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    record = json.loads(args.input.read_text().splitlines()[0])
    case = record["murdoku_case"]
    characters = case["characters"]
    who = characters[0]
    first = await make_environment(record).start()
    second = None
    try:

        async def act(env, text):
            return await env.step({"tool": "murdoku", "arguments": {"text": text}})

        await act(first, f"ACTION: mark {who} a1 b2")
        await act(first, f"ACTION: place {who} a1")
        state = await first.snapshot()
        second = await make_environment(record).start()
        await second.restore(state)
        assert await act(first, "ACTION: board") == await act(second, "ACTION: board")
        await act(first, f"ACTION: unplace {who}")
        assert await act(first, "ACTION: board") != await act(second, "ACTION: board")
        # Native scorer and renderer must agree. Derive the oracle transcript in
        # the trusted test process, never in the model's Python sandbox.
        raw = await first.step(
            {
                "tool": "run_python",
                "arguments": {"code": 'import os\nprint(os.path.exists("/murdoku"))'},
            }
        )
        assert raw.get("error") or "False" in str(raw), raw
        import subprocess

        code = """import json,sys
from murdoku_lab.core.instance import Case
from murdoku_lab.core.theme import canonical_theme,Theme
from murdoku_lab.evaluation.agents import OracleAgent
r=json.loads(sys.stdin.read());c=Case.from_json(r['murdoku_case']);t=Theme.from_json(r['murdoku_theme']) if r.get('murdoku_theme') else canonical_theme(c);print(OracleAgent(c,t).act('',[]))"""
        proc = subprocess.run(
            [str(root / ".venv/bin/python"), "-c", code],
            input=json.dumps(record),
            text=True,
            capture_output=True,
            cwd=root,
            check=True,
        )
        # Restore an empty scratchpad before an oracle answer, so earlier false
        # assertions cannot be silently credited as a correct complete solution.
        await first.close()
        first = await make_environment(record).start()
        result = await act(first, proc.stdout)
        assert (
            result["done"] and result["reward"] == 1 and result["score"]["solved"]
        ), result
        await second.close()
        second = await make_environment(record).start()
        result = await act(second, "ACTION: solve\nMURDERER: A")
        assert result["done"] and result["reward"] == 0, result
        await first.close()
        first = await make_environment(record).start()
        import re

        placements = dict(re.findall(r"ACTION: place (\S+) (\S+)", proc.stdout))
        verdict = re.search(r"ANSWER: (.+)", proc.stdout).group(1)
        result = await first.step(
            {
                "tool": "murdoku",
                "arguments": {
                    "action": "submit",
                    "placements": placements,
                    "murderer": verdict,
                },
            }
        )
        assert result["done"] and result["reward"] == 1, result
        await first.close()
        shaped = dict(
            record, murdoku_reward={"mode": "placement_shaped", "placement_weight": 0.2}
        )
        first = await make_environment(shaped).start()
        one_name, one_cell = next(iter(placements.items()))
        await first.step(
            {
                "tool": "murdoku",
                "arguments": {"action": "place", "person": one_name, "cell": one_cell},
            }
        )
        result = await first.finalize("turn_limit")
        assert (
            abs(result["reward"] - 0.2 / len(characters)) < 1e-12
            and not result["score"]["solved"]
        ), result
        print(
            json.dumps(
                {
                    "status": "MURDOKU_ENVIRONMENT_CONTRACT_PASS",
                    "case_id": record["prompt_uid"],
                    "snapshot_restore": True,
                    "oracle_reward": 1,
                    "verdict_only_reward": 0,
                    "model_python_key_access": False,
                }
            )
        )
    finally:
        await first.close()
        if second:
            await second.close()


asyncio.run(main())
