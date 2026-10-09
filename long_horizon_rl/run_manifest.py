"""Resolved configuration and credential-redacted run provenance."""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess

_SECRET_KEYS = {
    "token",
    "password",
    "passwd",
    "secret",
    "api_key",
    "access_token",
    "auth_token",
    "hf_token",
    "github_token",
    "authorization",
    "credentials",
    "client_secret",
}


def redact(value):
    if isinstance(value, dict):
        return {
            str(k): (
                "[REDACTED]"
                if str(k).lower() in _SECRET_KEYS
                or str(k)
                .lower()
                .endswith(("_api_key", "_access_token", "_password", "_secret"))
                else redact(v)
            )
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact(x) for x in value]
    if isinstance(value, str):
        value = re.sub(r"(?i)(https?://)[^/@\s]+:[^/@\s]+@", r"\1[REDACTED]@", value)
        value = re.sub(
            r"(?i)([?&](?:token|api_key|access_token)=)[^&\s]+", r"\1[REDACTED]", value
        )
        return re.sub(
            r"\b(?:gh[pousr]_[A-Za-z0-9_]+|github_pat_[A-Za-z0-9_]+|hf_[A-Za-z0-9]+)\b",
            "[REDACTED]",
            value,
        )
    return value


def sha256_file(path):
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(root):
    from .source_snapshot import git_revision as checked_revision

    return checked_revision(root)


def build_manifest(config, root):
    from omegaconf import OmegaConf

    resolved = OmegaConf.to_container(config, resolve=True)
    root = Path(root)
    pins = json.loads((root / "configs/runtime.json").read_text())
    # git archive snapshots intentionally have no .git; use recorded revision.
    frozen = root.parent / "project-revision.txt"
    revision = git_revision(root)
    if revision is None and frozen.exists():
        revision = frozen.read_text().strip()
    source_proof = root.parent / "source-fingerprint.json"
    source_metadata = (
        json.loads(source_proof.read_text()) if source_proof.exists() else None
    )
    dependencies = {}
    for name in (
        "torch",
        "transformers",
        "vllm",
        "ray",
        "transferqueue",
        "megatron-core",
        "transformer-engine",
        "sglang",
    ):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = None
    actor = resolved["actor_rollout_ref"]
    trainer = resolved["trainer"]
    paths = []
    for field in ("train_files", "val_files"):
        value = resolved["data"].get(field, [])
        for path in ([value] if isinstance(value, str) else value or []):
            paths.append({"role": field, "path": path, "sha256": sha256_file(path)})
    agent_path = actor["rollout"]["agent"].get("agent_loop_config_path")
    agents = None
    if agent_path and Path(agent_path).is_file():
        agents = OmegaConf.to_container(OmegaConf.load(agent_path), resolve=True)
    return redact(
        {
            "schema_version": 1,
            "compatibility_profile": trainer.get(
                "compatibility_profile", "VHD-HARDENED"
            ),
            "framework": {
                "repository": "https://github.com/volcengine/verl",
                "pinned_commit": pins["verl_commit"],
                "actual_commit": git_revision(root / "vendor/verl"),
            },
            "adapter": {
                "commit": revision,
                "frozen_source": frozen.exists() or source_metadata is not None,
                "source_fingerprint": (
                    source_metadata["sha256"] if source_metadata else None
                ),
                "source_origin": (
                    source_metadata["origin_kind"] if source_metadata else None
                ),
            },
            "dependency_lock": {
                "path": "configs/runtime.json",
                "sha256": sha256_file(root / "configs/runtime.json"),
                "actual_versions": dependencies,
            },
            "container_image_digest": os.getenv("LONG_HORIZON_CONTAINER_DIGEST"),
            "algorithm": {
                "trainer_mode": trainer["v1"]["trainer_mode"],
                "reduction": trainer.get("policy_reduction", "trajectory"),
                "statistical_unit": trainer.get("grpo_statistical_unit", "trajectory"),
            },
            "datasets": paths,
            "agent_config": agents,
            "resolved_config": resolved,
        }
    )


def write_manifest(config, root):
    manifest = build_manifest(config, root)
    if manifest["compatibility_profile"] != "VHD-HARDENED":
        raise ValueError("only the VHD-HARDENED compatibility profile is supported")
    target = Path(config.trainer.default_local_dir) / "resolved_run_manifest.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    with tmp.open("w") as handle:
        json.dump(manifest, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, target)
    return target
