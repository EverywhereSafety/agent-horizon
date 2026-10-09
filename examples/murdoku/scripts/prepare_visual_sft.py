"""Prepare visual SFT tensors and statistics using the official processor."""

import argparse
import json
from pathlib import Path
import torch
from transformers import AutoProcessor
from murdoku_demo.visual_sft import prepare_visual_example
from murdoku_demo.teacher_dataset import sft_example_from_row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--image-root", required=True)
    p.add_argument("--processor", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-length", type=int, default=131072)
    a = p.parse_args()
    processor = AutoProcessor.from_pretrained(a.processor)
    rows, stats = [], []
    seen = set()
    counts = {"input": 0, "selected": 0, "excluded": 0}
    with open(a.input) as source:
        for line in source:
            if not line.strip():
                continue
            counts["input"] += 1
            example = json.loads(line)
            if "query" in example:
                example = sft_example_from_row(
                    example, asset_root=a.image_root, view="vision"
                )
                if example is None:
                    counts["excluded"] += 1
                    continue
            counts["selected"] += 1
            uid = example["prompt_uid"]
            if uid in seen:
                raise ValueError(f"duplicate training query: {uid}")
            seen.add(uid)
            row = prepare_visual_example(example, processor, a.image_root, a.max_length)
            rows.append({"prompt_uid": uid, "model_inputs": row})
            stats.append(
                {
                    "prompt_uid": uid,
                    "tokens": row["input_ids"].numel(),
                    "supervised_tokens": int((row["labels"] != -100).sum()),
                    "images": len(row["image_grid_thw"]),
                }
            )
    if not rows:
        raise ValueError("empty vision training dataset")
    output = Path(a.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(rows, output)
    output.with_suffix(".summary.json").write_text(
        json.dumps(
            {
                "processor": a.processor,
                "max_length": a.max_length,
                "examples": stats,
                "selection": counts,
            },
            indent=2,
        )
    )
    print(f"Prepared {len(rows)} visual examples: {output}")


if __name__ == "__main__":
    main()
