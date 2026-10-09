"""Prompt-local selection with acknowledged permanent cancellation."""

import asyncio
import json
import os
import traceback
from pathlib import Path


class CancelledTrajectory(RuntimeError):
    pass


class CancellationLedger:
    def __init__(self, directory):
        self.directory = Path(directory)

    def path(self, uid):
        return self.directory / (uid.replace("/", "_") + ".cancelled.json")

    def check(self, uid):
        if self.path(uid).exists():
            raise CancelledTrajectory(uid)

    def mark(self, uid, reason):
        path = self.path(uid)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.loads(path.read_text()) if path.exists() else {}
        payload.update(trajectory_uid=uid, reason=reason, permanent=True)
        tmp = path.with_suffix(".tmp")
        with tmp.open("w") as f:
            json.dump(payload, f)
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(path)

    def acknowledge(self, uid, status):
        path = self.path(uid)
        payload = json.loads(path.read_text())
        payload["cleanup_acknowledgement"] = status
        tmp = path.with_suffix(".tmp")
        with tmp.open("w") as f:
            json.dump(payload, f)
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(path)


async def select_attempts(factory, uids, retain, predicate, ledger, cancel_timeout=15):
    """Return first valid completions; do not publish before all losers settle.

    The factory must propagate cancellation through generation and tool cleanup.
    Cancellation failures are fatal: returning a selected group would falsely
    acknowledge cleanup. Recovered tombstones are never scheduled again.
    """
    if not 0 < retain <= len(uids) or len(set(uids)) != len(uids):
        raise ValueError("invalid candidate identities")
    order = {}
    counter = 0

    async def run(uid):
        nonlocal counter
        ledger.check(uid)
        try:
            return await factory(uid)
        except Exception as exc:
            path = ledger.path(uid).with_name(
                ledger.path(uid).name.replace(".cancelled.json", ".failure.json")
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(
                    {
                        "trajectory_uid": uid,
                        "failure_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
                + "\n"
            )
            tmp.replace(path)
            raise
        finally:
            order[uid] = counter
            counter += 1

    tasks = {asyncio.create_task(run(uid)): uid for uid in uids}
    pending = set(tasks)
    selected = []
    chosen = set()
    failures = []

    async def settle(reason):
        losers = [task for task, uid in tasks.items() if uid not in chosen]
        for task in losers:
            ledger.mark(tasks[task], reason)
        for task in losers:
            if not task.done():
                task.cancel()
        if not losers:
            return
        done, unsettled = await asyncio.wait(losers, timeout=cancel_timeout)
        if unsettled:
            raise RuntimeError("trajectory cleanup was not acknowledged before timeout")
        for task in done:
            if not task.cancelled():
                exc = task.exception()
                # A previously failed attempt is already settled; cancellation
                # itself failing is not. Errors seen before selection are kept
                # distinct from those arising while stopping active work.
                if exc is not None and task in pending:
                    raise RuntimeError("trajectory cancellation failed") from exc
                ledger.acknowledge(
                    tasks[task],
                    "already_completed" if exc is None else "failed_and_cleaned",
                )
            else:
                ledger.acknowledge(tasks[task], "cancelled_and_cleaned")

    try:
        while pending and len(selected) < retain:
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )
            for task in sorted(done, key=lambda t: order[tasks[t]]):
                try:
                    value = task.result()
                except Exception as exc:
                    failures.append(exc)
                    continue
                if len(selected) < retain and predicate(value):
                    chosen.add(tasks[task])
                    selected.append(value)
        if len(selected) < retain:
            reasons = list(
                dict.fromkeys(type(exc).__name__ + ": " + str(exc) for exc in failures)
            )
            detail = (
                "; ".join(reasons[:3]) or "completed trajectories rejected by predicate"
            )
            raise RuntimeError(
                f"insufficient valid trajectories ({len(selected)}/{retain}): {detail}"
            ) from (failures[0] if failures else None)
        await settle("oversampling_redundant")
        for uid in chosen:
            ledger.check(uid)
        return selected
    except BaseException:
        chosen.clear()
        await settle("group_failed_or_cancelled")
        raise
