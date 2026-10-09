"""Episode-private cosine archive with pinned, locally cached text embeddings."""

import hashlib
import math
from functools import lru_cache


def normalized(vector):
    vector = [float(x) for x in vector]
    if not vector or not all(math.isfinite(x) for x in vector):
        raise ValueError("finite nonempty embedding required")
    norm = math.sqrt(sum(x * x for x in vector))
    if norm <= 0:
        raise ValueError("nonzero embedding required")
    return [x / norm for x in vector]


def vector_matches(memory, query, top_k, encoder):
    if not isinstance(query, str) or not query.strip():
        raise ValueError("nonempty archive query required")
    archive = memory.get("archival", {})
    index = memory.setdefault(
        "_archival_vector_index", {"encoder": encoder.identity, "entries": {}}
    )
    if index["encoder"] != encoder.identity:
        raise ValueError("archive embedding identity changed across resume")
    entries = index["entries"]
    for key in list(entries):
        if key not in archive:
            del entries[key]
    changed = [
        key
        for key, value in archive.items()
        if key not in entries
        or entries[key]["text_sha256"] != hashlib.sha256(value.encode()).hexdigest()
    ]
    if changed:
        vectors = encoder.encode([archive[key] for key in changed])
        if len(vectors) != len(changed):
            raise ValueError("unaligned embedding results")
        for key, vector in zip(changed, vectors):
            entries[key] = {
                "text_sha256": hashlib.sha256(archive[key].encode()).hexdigest(),
                "vector": normalized(vector),
            }
    q = normalized(encoder.encode([query])[0])
    scored = []
    for key, value in archive.items():
        v = entries[key]["vector"]
        if len(v) != len(q):
            raise ValueError("embedding dimension changed")
        scored.append((sum(a * b for a, b in zip(v, q)), key, value))
    scored.sort(key=lambda row: (-row[0], row[1]))
    return {
        "matches": {key: value for _, key, value in scored[:top_k]},
        "scores": {key: score for score, key, _ in scored[:top_k]},
        "done": False,
    }


class LocalTransformerEncoder:
    def __init__(self, model, revision, max_tokens=256):
        import torch
        from transformers import AutoTokenizer, AutoModel

        if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
            raise ValueError(
                "embedding revision must be an immutable 40-character commit"
            )
        self.identity = f"{model}@{revision}:mean-pool-normalized:{max_tokens}"
        self.max_tokens = max_tokens
        # No network/auth during an episode; stage assets before GPU allocation.
        self.tokenizer = AutoTokenizer.from_pretrained(
            model, revision=revision, local_files_only=True, trust_remote_code=False
        )
        self.model = (
            AutoModel.from_pretrained(
                model,
                revision=revision,
                local_files_only=True,
                trust_remote_code=False,
                dtype=torch.float32,
            )
            .to("cpu")
            .eval()
        )

    def encode(self, texts):
        import torch

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_tokens,
            return_tensors="pt",
        )
        with torch.inference_mode():
            hidden = self.model(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            return torch.nn.functional.normalize(pooled, dim=-1).tolist()


@lru_cache(maxsize=2)
def local_encoder(model, revision, max_tokens):
    return LocalTransformerEncoder(model, revision, max_tokens)
