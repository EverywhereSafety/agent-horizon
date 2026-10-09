"""Model-aware relative compute costs with native strict-cap partitions."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import math

_active_cost = ContextVar("long_horizon_model_cost", default=None)


@dataclass(frozen=True)
class ModelCost:
    quadratic: float
    linear: float

    def __post_init__(self):
        if any(
            not math.isfinite(x) or x < 0 for x in (self.quadratic, self.linear)
        ) or not (self.quadratic or self.linear):
            raise ValueError(
                "cost coefficients must be finite, nonnegative and nonzero"
            )

    def workload(self, lengths):
        return self.quadratic * lengths**2 + self.linear * lengths

    @classmethod
    def from_config(cls, config):
        if not isinstance(config, dict):
            config = config.to_dict()
        c = config.get("text_config", config)
        h = int(c["hidden_size"])
        layers = int(c["num_hidden_layers"])
        types = c.get("layer_types")
        if types is not None:
            if len(types) != layers:
                raise ValueError("layer_types must cover all layers")
            known = {"full_attention", "linear_attention", "sliding_attention"}
            if set(types) - known:
                raise ValueError(
                    "unknown attention type; supply explicit cost coefficients"
                )
            # Sliding attention cost is linear in length; approximate it with
            # its configured window instead of counting it as full attention.
            full = types.count("full_attention")
            sliding = types.count("sliding_attention")
            window = int(c.get("sliding_window") or 0)
            if sliding and window <= 0:
                raise ValueError("sliding attention requires its window")
        else:
            interval = c.get("full_attention_interval")
            full = layers // int(interval) if interval else layers
            sliding = 0
            window = 0
        expert_width = c.get("moe_intermediate_size")
        if expert_width is not None:
            ffn = int(expert_width) * int(c["num_experts_per_tok"]) + int(
                c.get("shared_expert_intermediate_size", 0)
            )
        else:
            ffn = int(c["intermediate_size"])
        if min(h, layers, ffn) <= 0:
            raise ValueError("positive model dimensions required")
        # Relative projection + active FFN + attention cost, NOT a hardware
        # latency model; excludes routing skew, communication and sparse CSA.
        heads = c.get("num_attention_heads")
        if heads is None:
            q_width = h
            attention_projection = 4 * h * h
        else:
            heads = int(heads)
            head_dim = int(c.get("head_dim") or h // heads)
            kv_heads = int(c.get("num_key_value_heads") or heads)
            if min(heads, head_dim, kv_heads) <= 0:
                raise ValueError("positive attention dimensions required")
            q_width = heads * head_dim
            kv_width = kv_heads * head_dim
            # Gated attention has an additional output-gate projection;
            # ordinary grouped-query attention uses separate Q, K, V and O.
            gated = bool(
                c.get("attn_output_gate", c.get("attention_output_gate", False))
            )
            attention_projection = h * ((3 if gated else 2) * q_width + 2 * kv_width)
        linear_layers = layers - full - sliding
        linear_projection = 4 * h * h
        gdn_fields = (
            "linear_num_key_heads",
            "linear_num_value_heads",
            "linear_key_head_dim",
            "linear_value_head_dim",
        )
        if linear_layers and all(c.get(k) is not None for k in gdn_fields):
            nk, nv, dk, dv = (int(c[k]) for k in gdn_fields)
            if min(nk, nv, dk, dv) <= 0:
                raise ValueError("positive linear attention dimensions required")
            key_width = nk * dk
            value_width = nv * dv
            linear_projection = h * (2 * key_width + 3 * value_width + 2 * nv)
            linear_projection += (2 * key_width + value_width) * int(
                c.get("linear_conv_kernel_dim", 4)
            )
            linear_projection += 2 * nv * dk * dv
        projection = (
            full + sliding
        ) * attention_projection + linear_layers * linear_projection
        return cls(
            2 * q_width * full,
            projection + layers * 3 * h * ffn + 2 * q_width * sliding * window,
        )


@contextmanager
def use_model_cost(cost):
    token = _active_cost.set(cost)
    try:
        yield
    finally:
        _active_cost.reset(token)


def rebalance_partitions(lengths, partitions, capacity, cost, max_iter=128):
    """Keep bin count, membership uniqueness and hard capacity; reduce max cost."""
    p = [list(x) for x in partitions]
    if sorted(i for b in p for i in b) != list(range(len(lengths))):
        raise ValueError("partitions must cover each sample exactly once")
    if any(not b or sum(lengths[i] for i in b) > capacity for b in p):
        raise ValueError("initial nonempty partitions must satisfy token capacity")

    def value(b):
        return sum(
            cost.quadratic * lengths[i] ** 2 + cost.linear * lengths[i] for i in b
        )

    for _ in range(max_iter):
        totals = [value(b) for b in p]
        high = max(range(len(p)), key=totals.__getitem__)
        best = None
        before = max(totals)
        for j in range(len(p)):
            if j == high:
                continue
            for i in p[high]:
                candidates = [None] if len(p[high]) > 1 else []
                candidates += p[j]
                for other in candidates:
                    a = [k for k in p[high] if k != i] + (
                        [] if other is None else [other]
                    )
                    b = [k for k in p[j] if k != other] + [i]
                    if any(sum(lengths[k] for k in row) > capacity for row in (a, b)):
                        continue
                    after = max(
                        [value(a), value(b)]
                        + [t for idx, t in enumerate(totals) if idx not in (high, j)]
                    )
                    if after < before and (best is None or after < best[0]):
                        best = (after, j, a, b)
        if best is None:
            break
        _, j, p[high], p[j] = best
    return p
