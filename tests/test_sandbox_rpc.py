"""Trusted fixtures exercise the production transport without requiring bubblewrap."""

import asyncio
import io
from pathlib import Path
import struct
import sys

import pytest

from long_horizon_rl.sandbox import (
    IsolatedEnvironment,
    SandboxRPCProtocolError,
    SandboxRPCDeadlineExceeded,
    SandboxError,
)
from long_horizon_rl.sandbox_rpc import encode_frame, read_frame, FrameError


class TrustedFixtureEnvironment(IsolatedEnvironment):
    def command(self):
        worker = (
            Path(__file__).resolve().parents[1] / "long_horizon_rl/sandbox_worker.py"
        )
        return [
            sys.executable,
            "-c",
            "import sys, runpy, resource; "
            "resource.setrlimit=lambda *args: None; "
            f"sys.argv=[{str(worker)!r}, {str(self.frame_limit)!r}]; "
            f"runpy.run_path({str(worker)!r}, run_name='__main__')",
        ]


def test_large_real_worker_roundtrip_snapshot_and_restore():
    async def run():
        record = dict(
            latent_dynamics="class LatentDynamics: pass\nclass Env:\n def __init__(self,d,v): self.value=v",
            initial_state_variable=0,
            database={"large": "界" * 400000},
            tool_schemas=[{"name": "echo"}],
            tool_implementations=[
                "def echo(text):\n return {'text': text, 'done': False}"
            ],
        )
        env = await TrustedFixtureEnvironment(record).start()
        try:
            text = "x" * (1024 * 1024)
            assert (await env.step({"tool": "echo", "arguments": {"text": text}}))[
                "text"
            ] == text
            state = await env.snapshot()
            assert len(state) > 65536
            assert (await env.restore(state))["restored"]
            with pytest.raises(SandboxError):
                await env.rpc({"op": "unknown"})
            assert (await env.rpc({"op": "context_update"}))["applied"] is False
        finally:
            await env.close()

    asyncio.run(run())


@pytest.mark.parametrize("size", [65536, 1024 * 1024, 16 * 1024 * 1024 - 12])
def test_frame_boundaries(size):
    value = {"text": "a" * size}
    wire = encode_frame(value)
    assert read_frame(io.BytesIO(wire)) == value
    with pytest.raises(FrameError):
        encode_frame(value, len(wire) - 5)
    with pytest.raises(FrameError):
        read_frame(io.BytesIO(wire[:-1]))


@pytest.mark.parametrize(
    "mode",
    ["oversized", "partial", "invalid", "envelope", "stderr", "timeout", "cancel"],
)
def test_protocol_failures_close_worker_and_stderr_is_bounded(mode):
    class Fixture(IsolatedEnvironment):
        def command(self):
            source = """import sys, struct, time
size=struct.unpack('!I',sys.stdin.buffer.read(4))[0]
sys.stdin.buffer.read(size)
"""
            if mode == "oversized":
                source += "sys.stdout.buffer.write(struct.pack('!I',999999999));sys.stdout.buffer.flush();time.sleep(5)"
            elif mode == "partial":
                source += "sys.stdout.buffer.write(struct.pack('!I',100)+b'{}');sys.stdout.buffer.flush()"
            elif mode == "invalid":
                source += "sys.stdout.buffer.write(struct.pack('!I',1)+b'x');sys.stdout.buffer.flush()"
            elif mode == "envelope":
                source += "sys.stdout.buffer.write(struct.pack('!I',2)+b'{}');sys.stdout.buffer.flush()"
            elif mode == "stderr":
                source += "sys.stderr.write('x'*200000);sys.stderr.flush();sys.stdout.buffer.write(struct.pack('!I',27)+b'{\"ok\":true,\"result\":{}}');sys.stdout.buffer.flush()"
            else:
                source += "sys.stdout.buffer.write(struct.pack('!I',100)+b'{');sys.stdout.buffer.flush();time.sleep(5)"
            return [sys.executable, "-c", source]

    async def run():
        env = Fixture({}, timeout=0.15 if mode == "timeout" else 2)
        if mode == "cancel":
            task = asyncio.create_task(env.start())
            await asyncio.sleep(0.1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            with pytest.raises(
                SandboxRPCDeadlineExceeded
                if mode == "timeout"
                else SandboxRPCProtocolError
            ):
                await env.start()
        assert env.process is None
        assert len(env._stderr_tail) <= 4096

    asyncio.run(run())


def test_oversized_request_closes_channel():
    async def run():
        record = dict(
            latent_dynamics="class LatentDynamics: pass\nclass Env:\n def __init__(self,d,v): pass",
            tool_implementations=[],
            tool_schemas=[],
            environment_rpc_max_bytes=1024,
        )
        env = await TrustedFixtureEnvironment(record).start()
        with pytest.raises(SandboxRPCProtocolError):
            await env.rpc({"op": "action", "large": "x" * 2048})
        assert env.process is None

    asyncio.run(run())
