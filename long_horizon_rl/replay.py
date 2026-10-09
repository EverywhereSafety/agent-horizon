import json
import os
from dataclasses import asdict
from pathlib import Path
from .contracts import EpisodeResult, Segment


class VersionedReplay:
    """CPU contract fixture. Upstream TransferQueue owns production distributed replay."""

    def __init__(self, max_lag=3):
        self.max_lag = max_lag
        self.version = 0
        self.ready = {}
        self.consumed = set()

    def publish_version(self, version, expected_replicas, acknowledgements):
        if (
            version != self.version + 1
            or not expected_replicas
            or set(acknowledgements) != set(expected_replicas)
        ):
            raise ValueError("incomplete publication barrier")
        if any(v != version for v in acknowledgements.values()):
            raise ValueError("replica version mismatch")
        self.version = version

    def add(self, group):
        if not group or len({r.prompt_uid for r in group}) != 1:
            raise ValueError("mixed group")
        uid = group[0].prompt_uid
        if uid in self.ready or uid in self.consumed:
            raise ValueError("duplicate prompt")
        if len({r.trajectory_uid for r in group}) != len(group):
            raise ValueError("duplicate trajectory")
        if any(r.termination != "terminal" for r in group):
            raise ValueError("nonterminal group")
        for r in group:
            for s in r.segments:
                s.validate()
            if (
                not r.dispatch_version
                <= r.min_served_version
                <= r.max_served_version
                <= self.version
            ):
                raise ValueError("invalid policy version lineage")
        self.ready[uid] = group

    def take(self):
        for uid in sorted(
            self.ready, key=lambda u: min(r.min_served_version for r in self.ready[u])
        ):
            group = self.ready.pop(uid)
            self.consumed.add(uid)
            if any(
                self.version - min(r.dispatch_version, r.min_served_version)
                > self.max_lag
                for r in group
            ):
                continue
            return group
        return None

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w") as f:
            json.dump(
                {
                    "max_lag": self.max_lag,
                    "version": self.version,
                    "consumed": sorted(self.consumed),
                    "ready": {u: [asdict(r) for r in g] for u, g in self.ready.items()},
                },
                f,
            )
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        obj = cls(d["max_lag"])
        obj.version = d["version"]
        obj.consumed = set(d["consumed"])
        for uid, group in d["ready"].items():
            for r in group:
                r["segments"] = [Segment(**s) for s in r["segments"]]
            obj.add([EpisodeResult(**r) for r in group])
        return obj
