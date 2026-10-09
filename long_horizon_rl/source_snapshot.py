"""Freeze a Git revision or an unversioned exported source tree honestly."""

import hashlib, json, os, shutil, subprocess, tarfile
from pathlib import Path

SOURCE_DIRS = (
    "long_horizon_rl",
    "scripts",
    "configs",
    "docs",
    "tests",
    "examples",
    "requirements",
)
SOURCE_FILES = (
    "README.md",
    "LICENSE",
    "LICENSE.txt",
    "pyproject.toml",
    "uv.lock",
    "requirements.txt",
    "MANIFEST.in",
    "Makefile",
    ".gitignore",
    ".gitmodules",
    ".python-version",
)


def git_revision(root, revision="HEAD"):
    root = Path(root).resolve()
    try:
        top = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        if Path(top).resolve() != root:
            return None
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "--verify", revision + "^{commit}"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def source_files(root):
    root = Path(root)
    files = []
    for name in SOURCE_FILES:
        p = root / name
        if p.is_file():
            files.append(p)
    for name in SOURCE_DIRS:
        for p in (root / name).rglob("*"):
            rel = p.relative_to(root)
            if (
                any(x in ("__pycache__", ".pytest_cache", ".git") for x in rel.parts)
                or p.suffix == ".pyc"
            ):
                continue
            if p.is_symlink():
                raise ValueError("export source symlink is not supported: " + str(rel))
            if p.is_file():
                files.append(p)
    return sorted(files, key=lambda p: str(p.relative_to(root)))


def fingerprint(root):
    root = Path(root)
    files = {}
    for p in source_files(root):
        h = hashlib.sha256()
        with p.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                h.update(chunk)
        files[p.relative_to(root).as_posix()] = h.hexdigest()
    digest = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {"algorithm": "sha256-file-tree-v1", "sha256": digest, "files": files}


def freeze(project, run, revision="HEAD", runtime=None, environment=None):
    project = Path(project).resolve()
    run = Path(run).resolve()
    source = run / "source"
    if source.exists():
        raise ValueError("refusing to overwrite existing validation source")
    runtime = Path(
        runtime or os.getenv("LONG_HORIZON_RUNTIME_ROOT") or project / "vendor/verl"
    ).resolve()
    environment = Path(
        environment or os.getenv("LONG_HORIZON_PYTHON_ENV") or project / ".venv"
    ).resolve()
    pins = json.loads((project / "configs/runtime.json").read_text())
    runtime_revision = git_revision(runtime)
    if runtime_revision != pins["verl_commit"]:
        raise ValueError(
            "runtime must be a verified checkout of the pinned public verl commit; run bootstrap or set LONG_HORIZON_RUNTIME_ROOT"
        )
    dirty = subprocess.check_output(
        ["git", "-C", str(runtime), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    )
    if dirty:
        raise ValueError("pinned runtime has tracked modifications")
    if not environment.is_dir():
        raise ValueError(
            "Python environment unavailable; run bootstrap or set LONG_HORIZON_PYTHON_ENV"
        )
    resolved = git_revision(project, revision)
    if resolved is None and revision != "HEAD":
        raise ValueError(
            "a requested Git revision cannot be resolved in an exported tree"
        )
    run.mkdir(parents=True, exist_ok=True)
    source.mkdir()
    if resolved:
        archive = run / "source.tar"
        subprocess.run(
            [
                "git",
                "-C",
                str(project),
                "archive",
                "--format=tar",
                "--output=" + str(archive),
                resolved,
            ],
            check=True,
        )
        with tarfile.open(archive) as handle:
            for member in handle.getmembers():
                path = Path(member.name)
                if (
                    path.is_absolute()
                    or ".." in path.parts
                    or not (member.isfile() or member.isdir())
                ):
                    raise ValueError("unsafe source archive member")
            if hasattr(tarfile, "data_filter"):
                handle.extractall(source, filter="data")
            else:
                handle.extractall(source)
        (run / "project-revision.txt").write_text(resolved + "\n")
        kind = "git_archive"
    else:
        for p in source_files(project):
            dest = source / p.relative_to(project)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)
        kind = "exported_tree"
    proof = fingerprint(source)
    proof.update(origin_kind=kind, git_commit=resolved)
    if not proof["files"]:
        raise ValueError("empty exported source")
    (run / "source-fingerprint.json").write_text(json.dumps(proof, indent=2) + "\n")
    (source / ".venv").symlink_to(environment, target_is_directory=True)
    (source / "vendor").mkdir(exist_ok=True)
    (source / "vendor/verl").symlink_to(runtime, target_is_directory=True)
    (run / "verl-revision.txt").write_text(runtime_revision + "\n")
    (run / "verl-worktree-status.txt").write_text(dirty)
    return source


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("project")
    p.add_argument("run")
    p.add_argument("revision", nargs="?", default="HEAD")
    args = p.parse_args()
    print(freeze(args.project, args.run, args.revision))
