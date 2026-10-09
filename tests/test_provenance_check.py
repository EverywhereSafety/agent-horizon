import unittest
from copy import deepcopy
from long_horizon_rl.contracts import Segment
from long_horizon_rl.diagnostics.provenance_check import check_segments


class ProvenanceCheck(unittest.TestCase):
    def fixture(self):
        first = Segment([1, 2])
        first.append([3], 1, [-0.2], 4, 5)
        first.append([8, 9], 0, [0.0, 0.0], -1)
        first.append([6], 1, [-0.3], 5, 5)
        cleared = Segment([7], turn_start=2)
        cleared.append([10], 1, [-0.4], 5, 6)
        rows = [
            dict(
                input_ids=p,
                output_ids=[t],
                log_probs=[v],
                served_version=a,
                max_served_version=b,
            )
            for p, t, v, a, b in [
                ([1, 2], 3, -0.2, 4, 5),
                ([1, 2, 3, 8, 9], 6, -0.3, 5, 5),
                ([7], 10, -0.4, 5, 6),
            ]
        ]
        return rows, [first, cleared]

    def test_tools_and_clear_preserve_conditions(self):
        rows, segments = self.fixture()
        self.assertEqual(check_segments(iter(rows), segments)["policy_tokens"], 3)

    def test_reject_rewritten_history(self):
        rows, segments = self.fixture()
        rows[1]["input_ids"][-1] = 99
        with self.assertRaisesRegex(ValueError, "conditioning"):
            check_segments(rows, segments)

    def test_reject_synthetic_policy_token(self):
        rows, segments = self.fixture()
        segments[0].response_mask[1] = 1
        with self.assertRaisesRegex(ValueError, "coverage"):
            check_segments(rows, segments)

    def test_reject_probability_or_version_rewrite(self):
        for field, value in [("log_probs", [-0.7]), ("max_served_version", 8)]:
            rows, segments = self.fixture()
            rows[0][field] = value
            with self.assertRaises(ValueError):
                check_segments(rows, deepcopy(segments))
