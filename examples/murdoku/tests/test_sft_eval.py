import unittest
from murdoku_demo.sft_eval import build_eval_plan, summarize_results


class EvaluationPlanTests(unittest.TestCase):
    def test_train_examples_were_used_in_sft_and_validation_stays_disjoint(self):
        rows = [
            {"prompt_uid": str(i), "split": "train" if i < 20 else "validation"}
            for i in range(30)
        ]
        selected = {str(i) for i in range(10)}
        plan = build_eval_plan(rows, selected)
        self.assertEqual(len(plan["tasks"]), 64)
        self.assertEqual(plan, build_eval_plan(list(reversed(rows)), selected))
        for task in plan["tasks"]:
            if task["split"] == "train":
                self.assertIn(task["prompt_uid"], selected)
            else:
                self.assertNotIn(task["prompt_uid"], selected)

    def test_invalid_attempts_are_not_model_failures(self):
        rows = [
            {"split": "train", "prompt_uid": "one", "status": "infrastructure_error"},
            {
                "split": "train",
                "prompt_uid": "one",
                "status": "complete",
                "strict_success": True,
                "reward": 1,
            },
        ]
        report = summarize_results(rows)["train"]
        self.assertEqual(report["valid_attempts"], 1)
        self.assertEqual(report["infrastructure_errors"], 1)
        self.assertEqual(report["questions_solved_at_least_once"], 1)
