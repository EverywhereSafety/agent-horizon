"""Deterministic operational memory; archival search uses lexical overlap."""

import re


def operate(memory, args, *, encoder=None):
    operation = args["operate"]
    tier, _, verb = operation.partition("_memory_")
    if tier not in ("global", "core", "archival"):
        raise ValueError("unknown memory tier")
    store = memory.setdefault(tier, {})
    key = (
        args.get("key")
        if tier == "core"
        else str(args.get("id", max([int(k) for k in store] + [-1]) + 1))
    )
    if verb in ("add", "update"):
        value = args.get("value")
        if not isinstance(value, str):
            raise ValueError("memory value must be a string")
        if key is None:
            raise ValueError("memory key required")
        key = str(key)
        if verb == "update" and key not in store:
            raise ValueError("memory key missing")
        store[key] = value
        return {"stored": key, "tier": tier, "done": False}
    if verb == "remove":
        store.pop(str(key), None)
        return {"removed": key, "done": False}
    if verb == "remove_all":
        store.clear()
        return {"cleared": tier, "done": False}
    if verb == "retrieve_all":
        return {"values": dict(store), "done": False}
    if verb == "retrieve":
        if tier == "core":
            return {"value": store.get(str(key)), "done": False}
        if encoder is not None:
            from .vector_memory import vector_matches

            k = args.get("top_k", 5)
            if type(k) is not int or k < 0:
                raise ValueError("invalid top_k")
            return vector_matches(memory, args.get("query", ""), k, encoder)
        words = set(re.findall(r"\w+", args.get("query", "").lower()))
        ranked = sorted(
            store.items(),
            key=lambda pair: (
                -len(words & set(re.findall(r"\w+", pair[1].lower()))),
                pair[0],
            ),
        )
        k = args.get("top_k", 5)
        if type(k) is not int or k < 0:
            raise ValueError("invalid top_k")
        return {"matches": dict(ranked[:k]), "done": False}
    raise ValueError("unknown memory operation")


def operate_notes(memory, args):
    """Episode-private title/content notes; body is returned only by read."""
    action = args.get("action")
    if action not in ("write", "read", "list", "delete"):
        raise ValueError("memory action must be write, read, list or delete")
    title = args.get("title")
    if action != "list" and (not isinstance(title, str) or not title.strip()):
        raise ValueError("nonempty memory title required")
    if action == "write" and not isinstance(args.get("content"), str):
        raise ValueError("memory content must be a string")
    store = memory.setdefault("notes", {})
    if action == "list":
        return {"titles": sorted(store), "done": False}
    if action == "write":
        store[title] = args["content"]
        return {"written": title, "done": False}
    if action == "read":
        if title not in store:
            return {"error": "memory title not found", "title": title, "done": False}
        return {"title": title, "content": store[title], "done": False}
    return {
        "deleted": title,
        "existed": store.pop(title, None) is not None,
        "done": False,
    }


class NotesStore:
    """Atomic per-episode file persistence, independent of visible context."""

    def __init__(self, path):
        from pathlib import Path
        import json

        self.path = Path(path)
        self.memory = json.loads(self.path.read_text()) if self.path.exists() else {}

    def operate(self, args):
        import json, os

        result = operate_notes(self.memory, args)
        if args["action"] in ("write", "delete"):
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temp = self.path.with_suffix(".tmp")
            with temp.open("w") as file:
                json.dump(self.memory, file, ensure_ascii=False)
                file.flush()
                os.fsync(file.fileno())
            temp.replace(self.path)
        return result
