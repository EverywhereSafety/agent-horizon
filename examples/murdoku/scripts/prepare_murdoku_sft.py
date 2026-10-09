"""Freeze unified JSONL and prepare native, untruncated SFT sequences."""

import argparse
import json
import shutil
import statistics
from pathlib import Path

from murdoku_demo.teacher_dataset import load_sft_examples, select_sft_examples
from murdoku_demo.sft_dataset import tokenize_example


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-length", type=int, default=65536)
    p.add_argument("--num-examples", type=int)
    p.add_argument("--seed", type=int, default=20261005)
    a = p.parse_args()
    import torch
    from transformers import AutoTokenizer

    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(a.dataset, out / "query_rollouts.jsonl")
    tokenizer = AutoTokenizer.from_pretrained(a.model)
    selection = {}
    accepted = load_sft_examples(
        out / "query_rollouts.jsonl", view="text", report=selection
    )
    examples = select_sft_examples(accepted, a.num_examples, a.seed)
    selected_uids = {row["prompt_uid"] for row in examples}
    unused_uids = sorted(
        row["prompt_uid"] for row in accepted if row["prompt_uid"] not in selected_uids
    )
    samples = []
    for example in examples:
        row = tokenize_example(example, tokenizer, a.max_length)
        row["input_ids"] = torch.tensor(row["input_ids"], dtype=torch.long)
        row["loss_mask"] = torch.tensor(row["loss_mask"], dtype=torch.long)
        samples.append(row)
    if not samples:
        raise ValueError("empty text training dataset")
    torch.save(samples, out / "train.pt")
    lengths = sorted(len(r["input_ids"]) for r in samples)
    report = {
        "examples": len(samples),
        "model": a.model,
        "max_length": a.max_length,
        "accepted_examples": len(accepted),
        "selection_seed": a.seed,
        "unused_train_prompt_uids": unused_uids,
        "selection": selection,
        "sequence_tokens": {
            "total": sum(lengths),
            "median": statistics.median(lengths),
            "max": max(lengths),
        },
        "supervised_tokens": sum(int(r["loss_mask"].sum()) for r in samples),
        "prompt_uids": [r["prompt_uid"] for r in samples],
        "template_kwargs": {"enable_thinking": True, "preserve_thinking": True},
        "loss": "assistant bodies and end-of-message; system/user/tool/header masked",
        "truncation": False,
        "epochs": 1,
    }
    (out / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "prompt_uids"}, indent=2))


if __name__ == "__main__":
    main()
