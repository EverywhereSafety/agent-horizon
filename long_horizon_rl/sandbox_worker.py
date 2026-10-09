"""Framed RPC worker. MUST be launched through the isolated client."""

import base64
import contextlib
import io
import json
import math
import pickle
import resource
import random
import sys
import types
from pathlib import Path
import importlib.util

# The isolated interpreter deliberately omits the caller's import paths.
_spec = importlib.util.spec_from_file_location(
    "sandbox_rpc", Path(__file__).with_name("sandbox_rpc.py")
)
_rpc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rpc)
frame_limit = _rpc.validate_limit(
    int(sys.argv[1]) if len(sys.argv) > 1 else _rpc.DEFAULT_MAX_FRAME_BYTES
)

resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
resource.setrlimit(resource.RLIMIT_FSIZE, (32 * 1024**2, 32 * 1024**2))
resource.setrlimit(resource.RLIMIT_NOFILE, (128, 128))
namespace = None
record = None


def handle(request):
    global namespace, record
    op = request["op"]
    if op == "init":
        record = request["record"]
        module = types.ModuleType("long_horizon_environment")
        sys.modules["long_horizon_environment"] = module
        # Existing environment snapshots refer to this original pickle module.
        sys.modules["vhd_environment"] = module
        namespace = module.__dict__
        exec(compile(record["latent_dynamics"], "<dynamics>", "exec"), namespace)
        namespace["env"] = namespace["Env"](
            namespace["LatentDynamics"](), record.get("initial_state_variable", 0)
        )
        namespace["database"] = record.get("database", {})
        for source in record["tool_implementations"]:
            exec(compile(source, "<tool>", "exec"), namespace)
        return {"initialized": True}
    if op == "snapshot":
        mode = record.get("environment_snapshot_mode", "instance_dict")
        env = namespace["env"]
        if mode == "hooks":
            if not callable(getattr(env, "snapshot_state", None)) or not callable(
                getattr(env, "restore_state", None)
            ):
                raise ValueError("hooks mode requires snapshot_state and restore_state")
            env_state = env.snapshot_state()
        elif mode == "instance_dict":
            if not hasattr(env, "__dict__"):
                raise ValueError(
                    "slot-only environment requires explicit snapshot hooks"
                )
            env_state = env.__dict__
        else:
            raise ValueError("unknown environment snapshot mode")
        names = ["database", *record.get("environment_state_globals", [])]
        if any(
            not isinstance(name, str)
            or not name.isidentifier()
            or name == "env"
            or name not in namespace
            for name in names
        ):
            raise ValueError("invalid environment_state_globals")
        state = {
            "env": env_state,
            "mode": mode,
            "globals": {name: namespace[name] for name in names},
            "python_random": random.getstate(),
            "random": (
                namespace.get("_rng_module").getstate()
                if "_rng_module" in namespace
                else None
            ),
        }
        return {
            "state": base64.b64encode(pickle.dumps(state)).decode(),
            "state_fields": sorted(env_state) if isinstance(env_state, dict) else [],
            "state_globals": sorted(state["globals"]),
            "snapshot_mode": mode,
        }
    if op == "restore":
        state = pickle.loads(base64.b64decode(request["state"]))
        mode = state.get("mode", "instance_dict")
        if mode != record.get("environment_snapshot_mode", "instance_dict"):
            raise ValueError("environment snapshot mode changed")
        if mode == "hooks":
            namespace["env"].restore_state(state["env"])
        else:
            namespace["env"].__dict__.clear()
            namespace["env"].__dict__.update(state["env"])
        namespace.update(state.get("globals", {}))
        if "python_random" in state:
            random.setstate(state["python_random"])
        if state["random"] is not None:
            namespace["_rng_module"].setstate(state["random"])
        return {"restored": True}
    if op == "context_update":
        env = namespace["env"]
        hook = getattr(env, "apply_context_update_friction", None)
        enabled = record.get("agent_info", {}).get(
            "apply_context_update_friction", False
        )
        if enabled and callable(hook):
            return {"applied": True, "result": hook()}
        return {"applied": False}
    if op == "action":
        action = request["action"]
        name = action["tool"]
        registered = {
            s.get("name", s.get("function", {}).get("name"))
            for s in record["tool_schemas"]
        }
        if name not in registered or name in ("operational_memory", "run_python"):
            raise ValueError("tool must be registered and environment-owned")
        args = action.get("arguments", {})
        if not isinstance(args, dict):
            raise ValueError("arguments must be an object")
        try:
            observation = namespace[name](**args)
        except Exception as exc:
            # A live tool may reject model arguments or fail in user code.
            # Return that failure to the policy; RPC/worker failures stay fatal.
            observation = {
                "error": f"{type(exc).__name__}: {exc}",
                "tool": name,
                "done": False,
            }
        env = namespace["env"]
        if not isinstance(observation, dict):
            observation = {"result": observation}
        if not observation.get("done") and hasattr(env, "next_turn"):
            transition = env.next_turn()
            if isinstance(transition, dict) and transition.get("done"):
                observation.update(transition)
        if observation.get("done"):
            value = observation.get("final_state_variable")
            if value is None:
                getter = getattr(env, "get_current_state_variable", None)
                if not callable(getter):
                    raise ValueError("terminal result requires final state or getter")
                value = getter()
            if "final_state_a" not in observation and callable(
                getattr(env, "get_current_a", None)
            ):
                observation["final_state_a"] = env.get_current_a()
            if not math.isfinite(float(value)):
                raise ValueError("nonfinite terminal state")
            observation["final_state_variable"] = value
            # Explicit initial smoke reward contract: state gain normalized by reference gain.
            # This is replaceable shaping, not historical reward equivalence.
            naive = float(record.get("naive_baseline", 0))
            optimum = float(
                record.get("analytical_optimal", record.get("baseline_score", 0))
            )
            denom = optimum - naive
            if denom <= 0:
                raise ValueError(
                    "positive reference gain required for normalized reward"
                )
            observation["reward"] = max(0.0, min(1.0, (float(value) - naive) / denom))
        return observation
    raise ValueError("unknown sandbox operation")


while True:
    try:
        request = _rpc.read_frame(sys.stdin.buffer, frame_limit)
        if request is None:
            break
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                result = handle(request)
            reply = {"ok": True, "result": result}
        except Exception as exc:
            reply = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        sys.stdout.buffer.write(_rpc.encode_frame(reply, frame_limit))
        sys.stdout.buffer.flush()
    except _rpc.FrameError:
        # A truncated or oversized frame invalidates this channel. Never retry
        # a possibly mutating action or emit a substitute success response.
        break
