"""Shared context feedback policy, independent of model, environment and rewards."""

import json


def memory_title_preview(memory, limit=32):
    titles = sorted(
        {str(key) for tier in ("notes", "core") for key in memory.get(tier, {})}
    )
    selected = [
        title[:128] + ("…" if len(title) > 128 else "") for title in titles[:limit]
    ]
    return {"titles": selected, "omitted": max(0, len(titles) - limit)}


def warning_notice(context_limit, trigger):
    return (
        f"Context budget reminder: the full input is approaching the "
        f"{trigger}-token history-clear trigger (window {context_limit} tokens). "
        "Use your available memory tools now to save important progress, "
        "deductions or code. Old complete interactions may be removed on "
        "a subsequent turn. The task, environment state and saved memory persist."
    )


def clear_notice(
    context_limit, trigger, removed_groups, removed_tokens, memory, title_limit=32
):
    index = memory_title_preview(memory, title_limit)
    return (
        f"Context clear: removed {removed_groups} oldest complete interactions "
        f"({removed_tokens} history tokens). The task, environment state and "
        f"saved memory persist. Full window: {context_limit} tokens; "
        f"clear trigger: {trigger} tokens. Saved memory title previews: "
        f"{json.dumps(index, ensure_ascii=False)}. Use memory list/read tools "
        "for full titles and contents. Note bodies are not included here."
    )
