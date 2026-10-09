import unittest
from long_horizon_rl.async_profile import validate_async_profile


def profile(warmup=1, span=4, single=False):
    return {
        "trainer": {
            "single_pass_unique_queries": single,
            "v1": {
                "trainer_mode": "long_horizon_separate_async",
                "separate_async": {"num_warmup_batches": warmup},
                "sampler": {"max_off_policy_threshold": span},
            },
        }
    }


class AsyncProfileTests(unittest.TestCase):
    def test_prefetch_with_version_headroom(self):
        validate_async_profile(profile())
        validate_async_profile(profile(0, 1))

    def test_prefetch_that_expires_before_use_is_rejected(self):
        for warmup, span in [(1, 1), (2, 2), (-1, 4), (True, 4), (0, 0)]:
            with self.subTest(warmup=warmup, span=span), self.assertRaises(ValueError):
                validate_async_profile(profile(warmup, span))

    def test_single_pass_must_not_overfetch(self):
        with self.assertRaises(ValueError):
            validate_async_profile(profile(single=True))
        validate_async_profile(profile(0, 4, True))
