import json, subprocess, tempfile, unittest
from pathlib import Path
from long_horizon_rl.source_snapshot import freeze, fingerprint, git_revision


class SourceSnapshots(unittest.TestCase):
    def git(self, root, *args):
        return subprocess.check_output(
            ["git", "-C", str(root), *args], text=True, stderr=subprocess.DEVNULL
        ).strip()

    def setup_tree(self, base):
        runtime = base / "runtime"
        runtime.mkdir()
        self.git(runtime, "init")
        (runtime / "code.py").write_text("pass\n")
        self.git(runtime, "add", "code.py")
        self.git(
            runtime,
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-m",
            "fixture",
        )
        project = base / "export"
        (project / "configs").mkdir(parents=True)
        (project / "long_horizon_rl").mkdir()
        (project / "long_horizon_rl/code.py").write_text("pass\n")
        (project / "configs/runtime.json").write_text(
            json.dumps({"verl_commit": git_revision(runtime)})
        )
        (project / "requirements").mkdir()
        (project / "requirements/sandbox.txt").write_text("numpy==2.0.0\n")
        (project / "requirements/sandbox.in").write_text("numpy\n")
        env = base / "env"
        env.mkdir()
        return project, runtime, env

    def test_export_freezes_content_without_fabricating_commit(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            project, runtime, env = self.setup_tree(base)
            source = freeze(project, base / "run", runtime=runtime, environment=env)
            proof = json.loads((base / "run/source-fingerprint.json").read_text())
            self.assertEqual(proof["origin_kind"], "exported_tree")
            self.assertIsNone(proof["git_commit"])
            self.assertFalse((base / "run/project-revision.txt").exists())
            self.assertEqual(fingerprint(source)["sha256"], proof["sha256"])
            self.assertTrue((source / "requirements/sandbox.txt").is_file())
            before = fingerprint(project)["sha256"]
            (project / "requirements/sandbox.in").write_text("numpy>=2\n")
            self.assertNotEqual(before, fingerprint(project)["sha256"])
            (project / "long_horizon_rl/code.py").write_text("changed\n")
            self.assertNotEqual(fingerprint(project)["sha256"], proof["sha256"])
            with self.assertRaises(ValueError):
                freeze(project, base / "run", runtime=runtime, environment=env)

    def test_parent_git_does_not_identify_nested_export_as_checkout(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            project, runtime, env = self.setup_tree(base)
            self.git(project, "init")
            self.git(project, "add", ".")
            self.git(
                project,
                "-c",
                "user.name=Fixture",
                "-c",
                "user.email=fixture@example.invalid",
                "commit",
                "-m",
                "fixture",
            )
            source = freeze(
                project, project / "outputs/run", runtime=runtime, environment=env
            )
            self.assertIsNone(git_revision(source))
            self.assertTrue((source / "requirements/sandbox.in").is_file())
            self.assertIn("requirements/sandbox.txt", fingerprint(source)["files"])
            self.assertEqual(
                (source.parent / "project-revision.txt").read_text().strip(),
                git_revision(project),
            )

    def test_runtime_pin_and_cleanliness_are_required(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            project, runtime, env = self.setup_tree(base)
            (runtime / "code.py").write_text("drift\n")
            with self.assertRaisesRegex(ValueError, "tracked modifications"):
                freeze(project, base / "run", runtime=runtime, environment=env)
