"""Native Qwen conversation tokenization with assistant-only supervision."""

import json
import copy
from pathlib import Path


def assistant_mask(input_ids, start_id, end_id, assistant_header):
    mask = [0] * len(input_ids)
    count = 0
    cursor = 0
    while cursor < len(input_ids):
        if input_ids[cursor] != start_id:
            cursor += 1
            continue
        begin = cursor + 1
        end = input_ids.index(end_id, begin)
        if start_id in input_ids[begin:end]:
            raise ValueError("nested chat delimiter")
        if input_ids[begin : begin + len(assistant_header)] == assistant_header:
            body = begin + len(assistant_header)
            mask[body : end + 1] = [1] * (end + 1 - body)
            count += 1
        cursor = end + 1
    return mask, count


def tokenize_example(example, tokenizer, max_length=65536):
    # Delimiters inside tool output would otherwise create false supervision spans.
    serialized = json.dumps(example, ensure_ascii=False)
    if "<|im_start|>" in serialized or "<|im_end|>" in serialized:
        raise ValueError("literal chat delimiter in example")
    messages = copy.deepcopy(example["messages"])
    # OpenAI response traces store arguments as JSON strings; native Qwen Jinja
    # expects mappings and serializes them into its XML tool-call syntax.
    for message in messages:
        for call in message.get("tool_calls") or []:
            function = call["function"]
            if isinstance(function["arguments"], str):
                function["arguments"] = json.loads(function["arguments"])
            if not isinstance(function["arguments"], dict):
                raise ValueError("tool arguments must be a JSON object")
    rendered = tokenizer.apply_chat_template(
        messages,
        tools=example["tools"],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=True,
        preserve_thinking=True,
    )
    ids = tokenizer.encode(rendered, add_special_tokens=False)
    if len(ids) > max_length:
        raise ValueError(
            f"{example['prompt_uid']}: {len(ids)} tokens exceeds {max_length}; no truncation"
        )
    header = tokenizer.encode("assistant\n", add_special_tokens=False)
    mask, count = assistant_mask(
        ids,
        tokenizer.convert_tokens_to_ids("<|im_start|>"),
        tokenizer.convert_tokens_to_ids("<|im_end|>"),
        header,
    )
    expected = sum(m["role"] == "assistant" for m in example["messages"])
    if count != expected or not any(mask):
        raise ValueError("assistant spans differ from source messages")
    # Ensure the official template retained every recorded thinking segment.
    for message in example["messages"]:
        reasoning = message.get("reasoning_content")
        if reasoning and reasoning.strip() not in rendered:
            raise ValueError("template dropped recorded reasoning")
    return {"prompt_uid": example["prompt_uid"], "input_ids": ids, "loss_mask": mask}


class TokenizedTeacherDataset:
    """veRL SFT custom dataset; tokenized once during CPU preparation."""

    def __init__(
        self, parquet_files, tokenizer, config, processor=None, max_samples=-1
    ):
        import torch

        paths = [parquet_files] if isinstance(parquet_files, str) else parquet_files
        self.samples = []
        for path in paths:
            self.samples.extend(
                torch.load(Path(path), map_location="cpu", weights_only=True)
            )
        if max_samples > 0:
            self.samples = self.samples[:max_samples]
        if config.pad_mode != "no_padding":
            raise ValueError("prepared teacher dataset requires no_padding")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        import torch

        row = self.samples[index]
        ids = row["input_ids"]
        return {
            "input_ids": ids,
            "position_ids": torch.arange(len(ids)),
            "loss_mask": row["loss_mask"],
        }
