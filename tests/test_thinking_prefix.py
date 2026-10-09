import ast, json
import pytest
from pathlib import Path
from long_horizon_rl.contracts import Config
from long_horizon_rl.continuation import ContinuationStore


def tokenizer_class(builder_class=None):
    tree = ast.parse(Path("long_horizon_rl/adapters/verl_v1.py").read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "QwenTokenizer"
    )

    class Builder:
        def __init__(self, tokenizer, chat_template_kwargs):
            self.kwargs = chat_template_kwargs

    ns = {"QwenContinuousTokenBuilder": builder_class or Builder, "json": json}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), "adapter-test", "exec"), ns)
    return ns["QwenTokenizer"]


def test_thinking_mode_controls_initial_and_subsequent_assistant_prefixes():
    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            return text

        def convert_tokens_to_ids(self, text):
            return 1

    cls = tokenizer_class()
    for enabled in [False, True]:
        t = cls(Tokenizer(), enable_thinking=enabled)
        assert t.builder.kwargs["enable_thinking"] is enabled
        prefix = "<|im_start|>assistant\n" + ("<think>\n" if enabled else "")
        assert t.observation("tool result").endswith(prefix)
        assert t.memory({"core": {"note": "value"}}).endswith(prefix)


def test_thinking_changes_resume_identity():
    assert (
        ContinuationStore("unused", {}, Config()).fingerprint
        != ContinuationStore("unused", {}, Config(enable_thinking=True)).fingerprint
    )


def test_native_builder_receives_tools_once_with_thinking():
    pytest.importorskip(
        "verl", reason="native tokenizer requires the pinned veRL runtime"
    )
    from verl.utils.tokenizer.continuous_token import QwenContinuousTokenBuilder

    class Tokenizer:
        def __init__(self):
            self.calls = []

        def encode(self, text, add_special_tokens=False):
            return [7]

        def convert_tokens_to_ids(self, text):
            return 8

        def apply_chat_template(self, messages, *, tools=None, **kwargs):
            self.calls.append((tools, kwargs))
            return [1, 2, 3]

    cls = tokenizer_class(QwenContinuousTokenBuilder)
    schemas = [
        {"name": "memory", "parameters": {"type": "object", "properties": {}}},
        {"name": "puzzle", "parameters": {"type": "object", "properties": {}}},
        {"name": "run_python", "parameters": {"type": "object", "properties": {}}},
    ]
    for thinking in [False, True]:
        tokenizer = Tokenizer()
        adapter = cls(tokenizer, schemas, enable_thinking=thinking)
        adapter.initial([{"role": "user", "content": "Puzzle"}])
        assert len(tokenizer.calls) == 1
        tools, kwargs = tokenizer.calls[0]
        assert tools == [{"type": "function", "function": s} for s in schemas]
        assert kwargs["enable_thinking"] is thinking
        assert "tools" not in adapter.builder.chat_template_kwargs
