"""Image-plus-conversation SFT preparation, independent of the puzzle setter."""

import copy
import base64
import io
import json
from pathlib import Path
from murdoku_demo.sft_dataset import assistant_mask


def resolve_messages(example, image_root):
    """Resolve local image references; never expose hidden grader fields."""
    messages = copy.deepcopy(example["messages"])
    root = Path(image_root).resolve()
    images = []
    if "<|im_start|>" in json.dumps(messages) or "<|im_end|>" in json.dumps(messages):
        raise ValueError("literal chat delimiter in conversation")
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if block.get("type") in ("image", "image_url"):
                    if message["role"] not in ("user", "tool"):
                        raise ValueError("images must occur in user or tool messages")
                    reference = (
                        block.get("image")
                        if block["type"] == "image"
                        else block["image_url"]["url"]
                    )
                    if reference.startswith("data:image/png;base64,"):
                        images.append(
                            io.BytesIO(
                                base64.b64decode(
                                    reference.split(",", 1)[1], validate=True
                                )
                            )
                        )
                        block.clear()
                        block.update(type="image", image=images[-1])
                        continue
                    relative = Path(reference)
                    path = (root / relative).resolve()
                    if relative.is_absolute() or root not in path.parents:
                        raise ValueError(
                            "image must be a relative path under image_root"
                        )
                    if not path.is_file():
                        raise ValueError(f"missing image: {relative}")
                    block.clear()
                    block.update(type="image", image=str(path))
                    images.append(path)
                elif block.get("type") != "text":
                    raise ValueError("only image and text content blocks are supported")
        for call in message.get("tool_calls") or []:
            arguments = call["function"]["arguments"]
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if not isinstance(arguments, dict):
                raise ValueError("tool arguments must be an object")
            call["function"]["arguments"] = arguments
    if not images:
        raise ValueError("visual SFT example has no image")
    return messages, images


def prepare_visual_example(example, processor, image_root, max_length=131072):
    """Preserve processor visual tensors; supervise assistant bodies only."""
    import torch
    from PIL import Image

    messages, paths = resolve_messages(example, image_root)
    tools = example.get(
        "tools",
        [
            {"type": "function", "function": schema}
            for schema in example.get("tool_schemas", [])
        ],
    )
    rendered = processor.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=True,
        preserve_thinking=True,
    )
    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    inputs = processor(
        text=[rendered], images=images, return_tensors="pt", padding=False
    )
    tokenizer = processor.tokenizer
    ids = inputs["input_ids"][0].tolist()
    if len(ids) > max_length:
        raise ValueError(
            f"{example['prompt_uid']}: {len(ids)} tokens exceeds {max_length}; no truncation"
        )
    mask, count = assistant_mask(
        ids,
        tokenizer.convert_tokens_to_ids("<|im_start|>"),
        tokenizer.convert_tokens_to_ids("<|im_end|>"),
        tokenizer.encode("assistant\n", add_special_tokens=False),
    )
    if count != sum(m["role"] == "assistant" for m in messages) or not any(mask):
        raise ValueError("assistant spans differ from source messages")
    for message in messages:
        reasoning = message.get("reasoning_content")
        if reasoning and reasoning.strip() not in rendered:
            raise ValueError("template dropped recorded reasoning")
    if "pixel_values" not in inputs or "image_grid_thw" not in inputs:
        raise ValueError("processor did not emit Qwen visual inputs")
    labels = inputs["input_ids"].clone()
    labels[0, ~torch.tensor(mask, dtype=torch.bool)] = -100
    # Explicitly keep visual positions out of the policy supervision mask.
    image_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    if any(m and token == image_id for token, m in zip(ids, mask)):
        raise ValueError("image tokens entered assistant supervision")
    inputs["labels"] = labels
    return dict(inputs)


class VisualTeacherDataset:
    """Prepared screenshots and exact conversations for veRL SFT no-padding."""

    def __init__(
        self, parquet_files, tokenizer, config, processor=None, max_samples=-1
    ):
        import torch

        if processor is None or config.pad_mode != "no_padding":
            raise ValueError("visual dataset requires a processor and no_padding")
        self.processor = processor
        paths = [parquet_files] if isinstance(parquet_files, str) else parquet_files
        self.samples = []
        for path in paths:
            self.samples.extend(torch.load(path, map_location="cpu", weights_only=True))
        if max_samples > 0:
            self.samples = self.samples[:max_samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        import torch
        from verl.models.transformers.qwen2_vl import get_rope_index

        inputs = self.samples[index]["model_inputs"]
        ids = inputs["input_ids"][0]
        vision_positions = get_rope_index(
            self.processor,
            input_ids=ids,
            image_grid_thw=inputs["image_grid_thw"],
            attention_mask=inputs["attention_mask"][0],
        )
        positions = torch.cat(
            (torch.arange(len(ids)).unsqueeze(0), vision_positions), dim=0
        )
        return {
            "input_ids": ids,
            "position_ids": positions,
            "loss_mask": (inputs["labels"][0] != -100).long(),
            "multi_modal_inputs": {
                k: v
                for k, v in inputs.items()
                if k not in ("input_ids", "attention_mask", "labels", "token_type_ids")
            },
        }
