"""Exercise public launch profiles without allocating GPUs or importing veRL."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class LauncherTests(unittest.TestCase):
    def launch(self, profile, overrides=(), **settings):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture = root / "arguments.json"
            executable = root / "capture-python"
            executable.write_text(
                f"#!{sys.executable}\nimport json, os, sys\n"
                'with open(os.environ["TEST_LAUNCH_CAPTURE"], "w") as f:\n'
                "    json.dump(sys.argv[1:], f)\n"
            )
            executable.chmod(0o755)
            env = {
                k: v for k, v in os.environ.items() if not k.startswith("LONG_HORIZON_")
            }
            env.update(
                LONG_HORIZON_PYTHON=str(executable),
                LONG_HORIZON_MODEL_PATH="/models/student",
                LONG_HORIZON_TEACHER_PATH="/models/teacher",
                LONG_HORIZON_TRAIN_FILE="/data/train.parquet",
                LONG_HORIZON_VAL_FILE="/data/validation.parquet",
                TEST_LAUNCH_CAPTURE=str(capture),
                **settings,
            )
            subprocess.run(
                ["bash", str(ROOT / "scripts" / profile), *overrides],
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            return json.loads(capture.read_text())

    def test_profiles_register_adapters_and_preserve_caller_precedence(self):
        for profile in ("train_smoke.sh", "train_long_horizon.sh", "train_opd.sh"):
            with self.subTest(profile=profile):
                args = self.launch(profile, ["trainer.total_training_steps=7"])
                self.assertEqual(args[:2], ["-m", "long_horizon_rl.train"])
                self.assertEqual(args[-1], "trainer.total_training_steps=7")
                self.assertNotIn(
                    "+actor_rollout_ref.model.override_config.attn_implementation=sdpa",
                    args,
                )
        args = self.launch("train_long_horizon.sh")
        self.assertIn("data.train_files=/data/train.parquet", args)
        self.assertIn("data.val_files=/data/validation.parquet", args)
        for key, expected in [
            ("actor_rollout_ref.actor.fsdp_config.param_offload", "False"),
            ("actor_rollout_ref.actor.fsdp_config.optimizer_offload", "False"),
            ("trainer.save_freq", "100"),
        ]:
            effective = [value for value in args if value.startswith(key + "=")][-1]
            self.assertEqual(effective, key + "=" + expected)

    def test_optional_model_backend_and_cpu_configuration(self):
        args = self.launch(
            "train_smoke.sh",
            LONG_HORIZON_ATTN_IMPLEMENTATION="sdpa",
            LONG_HORIZON_VLLM_PREFILL_BACKEND="triton",
            LONG_HORIZON_NUM_CPUS="8",
        )
        self.assertIn(
            "+actor_rollout_ref.model.override_config.attn_implementation=sdpa", args
        )
        self.assertIn(
            "+actor_rollout_ref.rollout.engine_kwargs.vllm.gdn_prefill_backend=triton",
            args,
        )
        self.assertIn("ray_kwargs.ray_init.num_cpus=8", args)
