import errno
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from long_horizon_rl.storage_budget import parse_lustre_available, checkpoint_space


class StorageBudgetTests(unittest.TestCase):
    def test_lustre_uses_user_limit_even_when_filesystem_is_large(self):
        output = "/private/host-storage 300* 300 300 - 40 1000 1000 -"
        self.assertEqual(parse_lustre_available(output), 0)
        self.assertEqual(
            parse_lustre_available(
                "/private/host-storage 200 300 300 - 40 1000 1000 -"
            ),
            100 * 1024,
        )
        self.assertEqual(
            parse_lustre_available(
                "/private/host-storage\n 200 300 300 - 40 1000 1000 -"
            ),
            100 * 1024,
        )
        self.assertIsNone(
            parse_lustre_available("/private/host-storage 200 0 0 - 40 0 0 -")
        )

    def test_quota_blocks_write_before_checkpoint_serialization(self):
        with (
            patch(
                "long_horizon_rl.storage_budget.shutil.disk_usage",
                return_value=SimpleNamespace(free=10**12),
            ),
            patch(
                "long_horizon_rl.storage_budget.shutil.which",
                return_value="/usr/bin/lfs",
            ),
            patch(
                "long_horizon_rl.storage_budget.subprocess.run",
                return_value=SimpleNamespace(
                    returncode=0,
                    stdout="/private/host-storage 200 300 300 - 40 1000 1000 -",
                ),
            ),
        ):
            with self.assertRaises(OSError) as raised:
                checkpoint_space(".", 101 * 1024)
            self.assertEqual(raised.exception.errno, errno.ENOSPC)
            self.assertEqual(
                checkpoint_space(".", 99 * 1024)["available_bytes"], 100 * 1024
            )
