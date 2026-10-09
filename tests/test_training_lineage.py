import json
import unittest
from long_horizon_rl.training_lineage import training_lineage


class TrainingLineageTests(unittest.TestCase):
    def test_query_coverage_counts_trajectories_and_ignores_padding(self):
        records = [
            json.dumps({"prompt_uid": uid})
            for uid in ("inventory", "inventory", "negotiation", "padding")
        ]
        result = training_lineage(
            ["a_0_0", "a_0_1", "b_1_0", "p_0_0"], [3, 2, 7, 0], records
        )
        self.assertEqual(result["prompt_uids"], ["inventory", "negotiation"])
        self.assertEqual(result["policy_trajectories"], 2)
        self.assertEqual(result["policy_tokens"], 12)

    def test_missing_identity_rejected(self):
        with self.assertRaises(KeyError):
            training_lineage(["a_0_0"], [1], ["{}"])
