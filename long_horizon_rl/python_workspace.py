"""Episode-local files, independent of model context and interpreter lifetime."""

import base64
import tempfile
import shutil
import os
from pathlib import Path


class PythonWorkspace:
    max_bytes = 16 * 1024**2
    max_files = 128
    max_directories = 1024

    def __init__(self):
        self._temp = tempfile.TemporaryDirectory(prefix="long-horizon-workspace-")
        self.root = Path(self._temp.name).resolve()
        self.closed = False

    def path(self, name):
        if not isinstance(name, str) or not name or Path(name).is_absolute():
            raise ValueError("path must be relative to /workspace")
        if ".." in Path(name).parts:
            raise ValueError("parent traversal is not allowed")
        target = self.root / name
        if any(
            (self.root / Path(*Path(name).parts[:i])).is_symlink()
            for i in range(1, len(Path(name).parts) + 1)
        ):
            raise ValueError("symbolic links are not supported")
        if not target.resolve().is_relative_to(self.root.resolve()):
            raise ValueError("path must stay inside /workspace")
        return target

    def snapshot(self):
        files = {}
        directories = []
        total = 0
        for p in sorted(self.root.rglob("*")):
            if p.is_symlink():
                raise ValueError("workspace snapshots cannot contain symbolic links")
            if not p.is_file():
                if not p.is_dir():
                    raise ValueError("workspace supports regular files only")
                directories.append(str(p.relative_to(self.root)))
                if len(directories) > self.max_directories:
                    raise ValueError("workspace exceeds directory snapshot budget")
                continue
            total += p.stat().st_size
            if total > self.max_bytes or len(files) >= self.max_files:
                raise ValueError("workspace exceeds 16 MiB / 128 file snapshot budget")
            files[str(p.relative_to(self.root))] = base64.b64encode(
                p.read_bytes()
            ).decode()
        return {"version": 1, "files": files, "directories": directories}

    def restore(self, snapshot):
        if not isinstance(snapshot, dict):
            raise ValueError("workspace snapshot must be an object")
        # Read legacy flat file maps as well as directory-aware snapshots.
        if isinstance(snapshot.get("files"), dict):
            if (
                set(snapshot) != {"version", "files", "directories"}
                or snapshot["version"] != 1
            ):
                raise ValueError("unsupported workspace snapshot")
            files, directories = snapshot["files"], snapshot["directories"]
            if not isinstance(directories, list):
                raise ValueError("directories must be a list")
        else:
            files, directories = snapshot, []
        if len(files) > self.max_files or len(directories) > self.max_directories:
            raise ValueError("workspace snapshot exceeds budget")
        decoded = {}
        for name, data in files.items():
            if not isinstance(data, str):
                raise ValueError("file data must be base64 text")
            decoded[name] = base64.b64decode(data, validate=True)
        if sum(map(len, decoded.values())) > self.max_bytes:
            raise ValueError("workspace snapshot exceeds budget")
        names = list(decoded) + directories
        for name in names:
            self.path(name)
            if name == "." or str(Path(name)) != name:
                raise ValueError("snapshot paths must be canonical relative paths")
        if len(set(names)) != len(names):
            raise ValueError("duplicate workspace paths")
        for name in names:
            if any(str(parent) in decoded for parent in Path(name).parents):
                raise ValueError("file cannot be the parent of another path")
        # Prepare the complete replacement before touching the live directory.
        # Keep its path stable, and roll back if installing the staged tree fails.
        temporary = Path(
            tempfile.mkdtemp(prefix="workspace-restore-", dir=self.root.parent)
        )
        staging = temporary / "new"
        backup = temporary / "old"
        try:
            staging.mkdir()
            for name in directories:
                (staging / name).mkdir(parents=True, exist_ok=True)
            for name, data in decoded.items():
                target = staging / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
            os.replace(self.root, backup)
            try:
                os.replace(staging, self.root)
            except BaseException:
                try:
                    os.replace(backup, self.root)
                except OSError as exc:
                    raise OSError(
                        f"workspace rollback failed; original files retained at {backup}"
                    ) from exc
                raise
            shutil.rmtree(backup, ignore_errors=True)
        finally:
            # Never remove the only surviving copy if rollback itself failed.
            if not backup.exists():
                shutil.rmtree(temporary, ignore_errors=True)

    def operate(self, args):
        try:
            return self._operate(args)
        except (
            FileNotFoundError,
            IsADirectoryError,
            NotADirectoryError,
            UnicodeError,
        ) as exc:
            return {"error": f"{type(exc).__name__}: {exc}", "done": False}

    def _operate(self, args):
        action = args.get("action")
        if action == "list":
            return {
                "files": [
                    {"path": name, "bytes": len(base64.b64decode(data))}
                    for name, data in self.snapshot()["files"].items()
                ],
                "done": False,
            }
        p = self.path(args.get("path"))
        if action == "read":
            if p.stat().st_size > self.max_bytes:
                raise ValueError("file exceeds budget")
            text = p.read_text()
            return {
                "content": text[:32768],
                "truncated": len(text) > 32768,
                "done": False,
            }
        if action == "delete":
            p.unlink()
        elif action in ("write", "replace"):
            content = args.get("content")
            if not isinstance(content, str):
                raise ValueError("content must be text")
            if action == "replace":
                old = args.get("old_text")
                if not isinstance(old, str) or not old:
                    raise ValueError("old_text must be nonempty text")
                original = p.read_text()
                if original.count(old) != 1:
                    raise ValueError("old_text must match exactly once")
                content = original.replace(old, content, 1)
            existing = self.snapshot()["files"]
            size = len(content.encode())
            others = {
                k: v for k, v in existing.items() if k != str(p.relative_to(self.root))
            }
            if (
                len(others) >= self.max_files
                or size + sum(len(base64.b64decode(v)) for v in others.values())
                > self.max_bytes
            ):
                raise ValueError("workspace exceeds 16 MiB / 128 files")
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
        else:
            raise ValueError("use write, replace, read, list or delete")
        return {"path": str(p.relative_to(self.root)), "action": action, "done": False}

    def close(self, archive_path=None):
        if self.closed:
            return
        try:
            if archive_path is not None:
                import json

                try:
                    saved = self.snapshot()
                except ValueError as exc:
                    saved = {"export_error": str(exc)}
                Path(archive_path).write_text(json.dumps(saved) + "\n")
        finally:
            self.closed = True
            self._temp.cleanup()
