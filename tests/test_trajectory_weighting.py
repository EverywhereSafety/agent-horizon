import unittest
from long_horizon_rl.trajectory_weighting import trajectory_weights


class TrajectoryWeightingTests(unittest.TestCase):
    def test_uneven_siblings_and_lengths(self):
        weights = trajectory_weights(
            ["prompt_0_0", "prompt_0_1", "prompt_1_0", "padding"], [1, 3, 2, 0]
        )
        self.assertEqual(weights, [0.125, 0.125, 0.25, 0])
        self.assertEqual(weights[0] + 3 * weights[1], 2 * weights[2])

    def test_minibatch_unbiased_scaling(self):
        weights = trajectory_weights(["prompt_0_0", "prompt_1_0"], [1, 3], 2)
        self.assertEqual(weights, [1, 1 / 3])

    def test_invalid_batch_fails(self):
        with self.assertRaises(ValueError):
            trajectory_weights(["bad"], [1])
        with self.assertRaises(ValueError):
            trajectory_weights(["padding"], [0])
