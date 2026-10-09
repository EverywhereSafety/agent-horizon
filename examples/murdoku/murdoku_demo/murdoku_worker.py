"""Trusted JSONL scratchpad worker. Model code never executes in this process."""

import contextlib, io, json, sys, resource

resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
resource.setrlimit(resource.RLIMIT_NOFILE, (128, 128))
sys.path.insert(0, "/murdoku")
from murdoku_lab.core.instance import Case
from murdoku_lab.core.theme import canonical_theme, Theme
from murdoku_lab.environment.state import Scratchpad, apply_action
from murdoku_lab.environment.answer_text import (
    extract_actions,
    absorb_placement,
    extract_verdict,
)
from murdoku_lab.environment.scoring import score_episode
from murdoku_lab.environment.actions import apply_structured_action
from murdoku_lab.environment.rewards import terminal_reward

pad = None
done = False
answer = None
turns = 0
allow_check = True
reward_config = None


def handle(request):
    global pad, done, answer, turns, allow_check, reward_config
    op = request["op"]
    if op == "init":
        case = Case.from_json(request["record"]["murdoku_case"])
        theme = (
            Theme.from_json(request["record"]["murdoku_theme"])
            if request["record"].get("murdoku_theme")
            else canonical_theme(case)
        )
        theme.validate(case)
        pad = Scratchpad(case, theme)
        allow_check = request["record"].get("allow_check", True)
        reward_config = request["record"].get("murdoku_reward")
        done = False
        answer = None
        turns = 0
        return {"initialized": True}
    if op == "snapshot":
        return {
            "state": {
                "case_id": pad.case.content_hash(),
                "theme": pad.theme.to_json(),
                "placed": pad.placed,
                "marks": {x: sorted(v) for x, v in pad.marks.items()},
                "done": done,
                "answer": answer,
                "turns": turns,
                "allow_check": allow_check,
            }
        }
    if op == "restore":
        state = request["state"]
        if (
            state["case_id"] != pad.case.content_hash()
            or state["allow_check"] != allow_check
            or state.get("theme", pad.theme.to_json()) != pad.theme.to_json()
        ):
            raise ValueError("Murdoku snapshot inputs changed")
        pad.placed = dict(state["placed"])
        pad.marks = {x: set(v) for x, v in state["marks"].items()}
        done = state["done"]
        answer = state["answer"]
        turns = state["turns"]
        return {"restored": True}
    if op == "finalize":
        if request["reason"] not in (
            "turn_limit",
            "generation_limit",
            "context_overflow",
            "immutable_context_overflow",
        ):
            raise ValueError("invalid stopping reason")
        score = score_episode(pad.case, pad.theme, pad, answer)
        done = True
        return {
            "done": True,
            "reward": terminal_reward(score, len(pad.case.characters), reward_config),
            "score": score,
        }
    if op == "context_update":
        return {"applied": False}
    if op == "action":
        if done:
            raise ValueError("episode already ended")
        action = request["action"]
        if action["tool"] != "murdoku":
            raise ValueError("unknown Murdoku tool")
        args = action.get("arguments", {})
        text = args.get("text")
        if "action" in args:
            result = apply_structured_action(pad, args, allow_check, reward_config)
            turns += 1
            done = result["done"]
            if done:
                answer = result["score"]["answer_raw"]
            return result
        if not isinstance(text, str):
            raise ValueError("Murdoku text required")
        actions = extract_actions(text)
        observations = []
        turns += 1
        for item in actions:
            verb = item.raw.split()[0].lower() if item.raw.split() else ""
            if verb == "python":
                observations.append(
                    "Use the separate run_python JSON tool for code execution."
                )
                continue
            if verb == "check" and not allow_check:
                observations.append("check is disabled")
                continue
            obs, submission = apply_action(pad, item.raw)
            observations.append(obs)
            if submission is not None:
                absorb_placement(pad, text)
                answer = extract_verdict(text) or submission
                done = True
                break
        if not actions:
            observations.append(
                "No ACTION line found. Use ACTION: place/mark/board/check/solve."
            )
        result = {"done": done, "observations": observations}
        if done:
            score = score_episode(pad.case, pad.theme, pad, answer)
            result.update(
                reward=terminal_reward(score, len(pad.case.characters), reward_config),
                final_state_variable=float(score["solved"]),
                score=score,
            )
        return result
    raise ValueError("unknown operation")


for line in sys.stdin:
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            result = handle(json.loads(line))
        print(json.dumps({"ok": True, "result": result}, allow_nan=False), flush=True)
    except Exception as exc:
        print(
            json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}),
            flush=True,
        )
