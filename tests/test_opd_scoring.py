import asyncio
import unittest
from types import SimpleNamespace
from long_horizon_rl.adapters.opd_scoring import score_earlier_segments


def output(i):
    return SimpleNamespace(prompt_ids=[i], response_ids=[i + 10], extra_fields={})


class OPDScoringTests(unittest.IsolatedAsyncioTestCase):
    async def test_all_earlier_contexts_scored_with_bounded_concurrency(self):
        groups = [[output(i) for i in range(7)], [output(i) for i in range(7, 12)]]
        active = peak = 0
        seen = []

        async def score(o, **kwargs):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            try:
                self.assertEqual(kwargs["prompt_ids"], o.prompt_ids)
                self.assertEqual(kwargs["response_ids"], o.response_ids)
                await asyncio.sleep(0.001)
                o.extra_fields.update(
                    teacher_logprobs=[-0.5], teacher_ids=o.response_ids
                )
                seen.append(o.prompt_ids[0])
            finally:
                active -= 1

        await score_earlier_segments(
            SimpleNamespace(_compute_teacher_logprobs=score), groups, {}, 3
        )
        self.assertEqual(set(seen), set(range(12)) - {6, 11})
        self.assertEqual(peak, 3)
        self.assertEqual(active, 0)
        self.assertEqual(groups[0][-1].extra_fields, {})

    async def test_failure_settles_all_outstanding_tasks(self):
        active = 0

        async def score(o, **kwargs):
            nonlocal active
            active += 1
            try:
                if o.prompt_ids == [0]:
                    await asyncio.sleep(0.001)
                    return  # Missing teacher fields must fail publication.
                await asyncio.Event().wait()
            finally:
                active -= 1

        with self.assertRaisesRegex(ValueError, "Missing teacher scores"):
            await score_earlier_segments(
                SimpleNamespace(_compute_teacher_logprobs=score),
                [[output(i) for i in range(6)]],
                {},
                3,
            )
        self.assertEqual(active, 0)


import unittest
from importlib.util import find_spec
from unittest.mock import patch
import torch
from long_horizon_rl.adapters.opd_padding import padding_safe_opd_loss


class OPDPaddingTests(unittest.TestCase):
    def test_synthetic_microbatch_has_zero_loss_and_zero_gradient(self):
        values = torch.tensor([0.2, -0.4], requires_grad=True)
        loss, metrics = padding_safe_opd_loss(
            {"log_probs": values},
            {"response_mask": torch.zeros(2)},
            config=None,
            distillation_config=None,
        )
        loss.backward()
        self.assertEqual(loss.item(), 0.0)
        self.assertEqual(metrics, {})
        torch.testing.assert_close(values.grad, torch.zeros_like(values))

    @unittest.skipUnless(
        find_spec("verl"), "native OPD loss requires the pinned veRL runtime"
    )
    def test_mixed_real_and_padding_preserves_native_loss(self):
        data = {"response_mask": torch.tensor([[1.0, 0.0], [0.0, 0.0]])}
        output = {"log_probs": torch.zeros(2, 2, requires_grad=True)}
        expected = (torch.tensor(0.3), {"native": 1})
        with patch(
            "verl.trainer.distillation.losses.distillation_ppo_loss",
            return_value=expected,
        ) as native:
            actual = padding_safe_opd_loss(
                output, data, config="actor", distillation_config="teacher"
            )
        self.assertIs(actual, expected)
        native.assert_called_once_with(
            config="actor",
            distillation_config="teacher",
            model_output=output,
            data=data,
            dp_group=None,
        )
