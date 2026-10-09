import difflib
import json
import re


def parse_action(text, schemas=None):
    """Parse JSON tool calls and Qwen XML calls; ignore the reasoning prefix."""
    text = text.replace("<|im_end|>", "").replace("<|endoftext|>", "")
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    match = re.search(r"<tool_call>(.*?)</tool_call>", text, re.S)
    if match:
        text = match.group(1).strip()
    text = text.strip()
    fence = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.S | re.I)
    if fence:
        text = fence.group(1).strip()
    xml = re.fullmatch(r"<function=([^>]+)>(.*?)</function>", text, re.S)
    if xml:
        name = xml.group(1)
        args = {}
        for key, value in re.findall(
            r"<parameter=([^>]+)>(.*?)</parameter>", xml.group(2), re.S
        ):
            try:
                args[key] = json.loads(value.strip())
            except ValueError:
                args[key] = value.strip()
        action = {"tool": name, "arguments": args}
    else:
        action = json.loads(text)
        if not isinstance(action, dict):
            raise ValueError("tool call must be an object")
        if "name" in action:
            action = {"tool": action["name"], "arguments": action.get("arguments", {})}
        if isinstance(action.get("arguments"), str):
            action["arguments"] = json.loads(action["arguments"])
    if not isinstance(action.get("tool"), str):
        raise ValueError("tool name required")
    if schemas:
        names = [s.get("name", s.get("function", {}).get("name")) for s in schemas]
        name = action["tool"]
        if name not in names:
            normalize = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
            matches = [n for n in names if normalize(n) == normalize(name)]
            if len(matches) != 1:
                matches = difflib.get_close_matches(name, names, n=1, cutoff=0.85)
            if not matches:
                raise ValueError("unknown tool name")
            action["tool"] = matches[0]
        args = action.get("arguments", {})
        if not isinstance(args, dict):
            raise ValueError("arguments must be an object")
        schema = next(
            s
            for s in schemas
            if s.get("name", s.get("function", {}).get("name")) == action["tool"]
        )
        schema = schema.get("function", schema).get("parameters", {})
        for key in schema.get("required", []):
            if key not in args:
                raise ValueError(f"missing argument {key}")
    return action


def parse_actions(text, schemas=None):
    """Parse every native call in order, validating the whole batch before execution."""
    visible = text.rsplit("</think>", 1)[-1] if "</think>" in text else text
    calls = re.findall(r"<tool_call>(.*?)</tool_call>", visible, re.S)
    if not calls:
        return [parse_action(visible, schemas)]
    if visible.count("<tool_call>") != len(calls) or visible.count(
        "</tool_call>"
    ) != len(calls):
        raise ValueError("incomplete tool call batch")
    return [parse_action(call, schemas) for call in calls]
