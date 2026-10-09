"""Check launcher contracts without allocating GPUs or importing veRL."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/train_visual_sft.sh"


class SFTLauncherTests(unittest.TestCase):
    def launch(self, **settings):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stub = root / "torchrun"
            stub.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$CAPTURE"\n')
            stub.chmod(0o755)
            env = {
                k: v for k, v in os.environ.items() if not k.startswith("VISUAL_SFT_")
            }
            env.update(
                PATH=f'{root}:{env["PATH"]}',
                CAPTURE=str(root / "args"),
                VISUAL_SFT_MODEL="/model",
                VISUAL_SFT_DATA="/data.pt",
                VISUAL_SFT_OUTPUT="/checkpoint",
                **settings,
            )
            result = subprocess.run(
                ["bash", str(SCRIPT)], env=env, capture_output=True, text=True
            )
            args = (
                (root / "args").read_text().splitlines()
                if (root / "args").exists()
                else []
            )
            return result, args

    def test_eight_gpus_get_eight_example_batch_and_verl_fused_loss(self):
        result, args = self.launch(VISUAL_SFT_GPUS="8")
        self.assertEqual(result.returncode, 0, result.stderr)
        for value in [
            "verl.trainer.sft_trainer",
            "--nproc_per_node=8",
            "data.train_batch_size=8",
            "model.use_fused_kernels=True",
            "model.fused_kernel_options.impl_backend=liger",
            "trainer.resume_mode=auto",
            "trainer.save_freq=-1",
        ]:
            self.assertIn(value, args)

    def test_accumulation_and_optional_backend_and_save_interval(self):
        result, args = self.launch(
            VISUAL_SFT_GPUS="4",
            VISUAL_SFT_BATCH="16",
            VISUAL_SFT_FUSED_BACKEND="torch",
            VISUAL_SFT_SAVE_FREQ="50",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("data.train_batch_size=16", args)
        self.assertIn("model.fused_kernel_options.impl_backend=torch", args)
        self.assertIn("trainer.save_freq=50", args)

    def test_incompatible_batch_fails_before_torchrun(self):
        for batch in ["4", "9", "0"]:
            with self.subTest(batch=batch):
                result, args = self.launch(VISUAL_SFT_GPUS="8", VISUAL_SFT_BATCH=batch)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(args, [])
