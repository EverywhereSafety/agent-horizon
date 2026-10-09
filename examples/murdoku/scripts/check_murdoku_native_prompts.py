"""Preflight actual native Qwen/tool rendering before allocating trainer GPUs."""

import argparse, ast, json, sys
from pathlib import Path
from transformers import AutoTokenizer
from verl.utils.tokenizer.continuous_token import QwenContinuousTokenBuilder

p = argparse.ArgumentParser()
p.add_argument("--input", type=Path, required=True)
p.add_argument("--model", type=Path, required=True)
a = p.parse_args()
tree = ast.parse(Path("long_horizon_rl/adapters/verl_v1.py").read_text())
cls = next(
    n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "QwenTokenizer"
)
namespace = {"QwenContinuousTokenBuilder": QwenContinuousTokenBuilder, "json": json}
exec(
    compile(
        ast.Module(body=[cls], type_ignores=[]), "native-tokenizer-preflight", "exec"
    ),
    namespace,
)
tokenizer = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
lengths = []
if a.input.suffix == ".parquet":
    import pandas as pd

    records = [json.loads(r) for r in pd.read_parquet(a.input)["record_json"]]
else:
    records = [json.loads(l) for l in a.input.read_text().splitlines()]
for record in records:
    adapter = namespace["QwenTokenizer"](
        tokenizer, record["tool_schemas"], enable_thinking=True
    )
    ids = adapter.initial(record["messages"])
    lengths.append(len(ids))
    assert tokenizer.decode(ids[-12:]).endswith(
        "<think>\n"
    ), "native thinking prefix missing"
    assert (
        len(ids) <= 8192
    ), f'Initial tool-rendered prompt exceeds cap: {record["prompt_uid"]}'
print(
    json.dumps(
        {
            "status": "NATIVE_MURDOKU_PROMPTS_PASS",
            "records": len(lengths),
            "maximum_prompt_tokens": max(lengths),
            "minimum_prompt_tokens": min(lengths),
            "tools": "native schema passed once",
            "thinking": True,
        }
    ),
    flush=True,
)
