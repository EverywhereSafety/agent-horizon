import unittest
from importlib.util import find_spec
import torch
from long_horizon_rl.diagnostics.model_cost import (
    ModelCost,
    rebalance_partitions,
    use_model_cost,
    _active_cost,
)


class ModelCostTests(unittest.TestCase):
    def test_hybrid_and_moe_active_width(self):
        c = {
            "hidden_size": 10,
            "num_hidden_layers": 4,
            "layer_types": ["linear_attention"] * 3 + ["full_attention"],
            "moe_intermediate_size": 20,
            "num_experts_per_tok": 2,
            "shared_expert_intermediate_size": 10,
        }
        cost = ModelCost.from_config({"text_config": c})
        self.assertEqual(cost.quadratic, 20)
        self.assertEqual(cost.linear, 4 * (400 + 1500))
        c["layer_types"] = ["full_attention"] * 4
        self.assertEqual(ModelCost.from_config(c).quadratic, 80)

    def test_explicit_head_width_and_grouped_query_projection(self):
        c = {
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "intermediate_size": 32,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "head_dim": 8,
        }
        cost = ModelCost.from_config(c)
        self.assertEqual(cost.quadratic, 2 * 32 * 2)
        self.assertEqual(cost.linear, 2 * (16 * (2 * 32 + 2 * 8) + 3 * 16 * 32))
        c["attn_output_gate"] = True
        self.assertEqual(ModelCost.from_config(c).linear - cost.linear, 2 * 16 * 32)

    def test_hybrid_gdn_projection_uses_linear_attention_dimensions(self):
        c = {
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "intermediate_size": 32,
            "num_attention_heads": 2,
            "head_dim": 8,
            "layer_types": ["linear_attention", "full_attention"],
            "linear_num_key_heads": 1,
            "linear_num_value_heads": 2,
            "linear_key_head_dim": 8,
            "linear_value_head_dim": 8,
            "linear_conv_kernel_dim": 4,
        }
        cost = ModelCost.from_config(c)
        self.assertEqual(cost.quadratic, 32)
        expected_gdn = 16 * (2 * 8 + 3 * 16 + 2 * 2) + (2 * 8 + 16) * 4 + 2 * 2 * 8 * 8
        self.assertEqual(cost.linear, 4 * 16 * 16 + expected_gdn + 2 * 3 * 16 * 32)

    def test_capacity_membership_and_nonincreasing_maximum(self):
        cost = ModelCost(1.0, 0.0)
        lengths = [10, 8, 6, 4, 2]
        initial = [[0, 1], [2, 3, 4]]
        result = rebalance_partitions(lengths, initial, 20, cost)
        self.assertEqual(sorted(i for b in result for i in b), list(range(5)))
        self.assertTrue(all(b and sum(lengths[i] for i in b) <= 20 for b in result))

        def peak(p):
            return max(sum(lengths[i] ** 2 for i in b) for b in p)

        self.assertLess(peak(result), peak(initial))
        with self.assertRaises(ValueError):
            rebalance_partitions(lengths, initial, 17, cost)
        with self.assertRaises(ValueError):
            ModelCost(float("nan"), 1.0)

    def test_scoped_cost_restores_on_failure(self):
        self.assertIsNone(_active_cost.get())
        with self.assertRaises(RuntimeError):
            with use_model_cost(ModelCost(1.0, 2.0)):
                self.assertIsNotNone(_active_cost.get())
                raise RuntimeError()
        self.assertIsNone(_active_cost.get())

    @unittest.skipUnless(
        find_spec("verl"), "native actor partitioning requires the pinned veRL runtime"
    )
    def test_worker_keeps_native_separate_async_detach_contract(self):
        from verl.experimental.separation.engine_workers import DetachActorWorker
        from long_horizon_rl.diagnostics.model_cost_v1 import (
            ModelAwareActorRolloutRefWorker,
        )

        self.assertTrue(issubclass(ModelAwareActorRolloutRefWorker, DetachActorWorker))

    @unittest.skipUnless(
        find_spec("verl"), "native actor partitioning requires the pinned veRL runtime"
    )
    def test_native_partition_preserves_metadata_and_order_inverse(self):
        from tensordict import TensorDict
        from verl.utils.tensordict_utils import (
            nested_tensor_from_tensor_list,
            assign_non_tensor,
        )
        from verl.workers.engine.utils import prepare_micro_batches
        from long_horizon_rl.diagnostics.model_cost_v1 import install_partition_adapter

        install_partition_adapter()
        data = TensorDict(
            {
                "input_ids": nested_tensor_from_tensor_list(
                    [torch.ones(n, dtype=torch.long) for n in [10, 8, 6, 4, 2]],
                    ragged_idx=1,
                ),
                "row_id": torch.arange(5),
            },
            batch_size=5,
        )
        assign_non_tensor(
            data, use_dynamic_bsz=True, max_token_len_per_gpu=20, global_batch_size=5
        )
        with use_model_cost(ModelCost(1.0, 0.0)):
            batches, indices = prepare_micro_batches(data, same_micro_num_in_dp=False)
        self.assertEqual(sorted(i for b in indices for i in b), list(range(5)))
        for b, idx in zip(batches, indices):
            self.assertEqual(b["row_id"].tolist(), idx)
            self.assertEqual(b["global_batch_size"], 5)
            self.assertLessEqual(b["input_ids"].offsets().diff().sum(), 20)
