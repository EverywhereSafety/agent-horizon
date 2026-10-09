import json
import tempfile
from pathlib import Path
import unittest
from omegaconf import OmegaConf
from long_horizon_rl.run_manifest import redact, write_manifest


class ManifestTests(unittest.TestCase):
    def test_redacts_credentials_but_preserves_token_limits_and_topology(self):
        value = {
            "hf_token": "secret",
            "tokenizer": "qwen",
            "max_new_tokens": 16384,
            "endpoint": "https://user:password@host/path?access_token=secret&x=2",
            "nested": [{"api_key": "secret", "n_gpus_per_node": 4}],
        }
        result = redact(value)
        self.assertEqual(result["hf_token"], "[REDACTED]")
        self.assertEqual(result["max_new_tokens"], 16384)
        self.assertEqual(result["nested"][0]["n_gpus_per_node"], 4)
        self.assertNotIn("secret", json.dumps(result))
        self.assertNotIn("password", result["endpoint"])

    def test_frozen_source_revision_input_hash_and_resolved_config(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "source"
            (root / "configs").mkdir(parents=True)
            (root / "configs/runtime.json").write_text(
                json.dumps({"verl_commit": "pin", "transferqueue_commit": "tq"})
            )
            (root.parent / "project-revision.txt").write_text("frozen-revision")
            dataset = root / "data.parquet"
            dataset.write_bytes(b"fixture")
            cfg = OmegaConf.create(
                {
                    "trainer": {
                        "v1": {"trainer_mode": "external_test_trainer"},
                        "default_local_dir": str(root / "ckpt"),
                    },
                    "data": {"train_files": str(dataset), "val_files": []},
                    "actor_rollout_ref": {"rollout": {"agent": {}}},
                }
            )
            path = write_manifest(cfg, root)
            manifest = json.loads(path.read_text())
            self.assertEqual(manifest["adapter"]["commit"], "frozen-revision")
            self.assertEqual(len(manifest["datasets"][0]["sha256"]), 64)
            self.assertEqual(
                manifest["algorithm"]["trainer_mode"], "external_test_trainer"
            )
            cfg.trainer.compatibility_profile = "VHD-EXACT-2026-09"
            with self.assertRaises(ValueError):
                write_manifest(cfg, root)
