"""Bounded, length-prefixed JSON frames for the isolated environment process."""

import json
import struct

DEFAULT_MAX_FRAME_BYTES = 16 * 1024**2
MAX_FRAME_BYTES = 64 * 1024**2
HEADER_BYTES = 4


class FrameError(ValueError):
    pass


def validate_limit(value):
    if type(value) is not int or not 0 < value <= MAX_FRAME_BYTES:
        raise ValueError("environment_rpc_max_bytes must be an integer in 1..64 MiB")
    return value


def frame_size(header, limit):
    size = struct.unpack("!I", header)[0]
    if not 0 < size <= limit:
        raise FrameError(f"RPC frame size {size} exceeds allowed range 1..{limit}")
    return size


def encode_frame(value, limit=DEFAULT_MAX_FRAME_BYTES):
    try:
        body = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        raise FrameError("RPC payload is not serializable JSON") from exc
    if not 0 < len(body) <= limit:
        raise FrameError(f"RPC frame size {len(body)} exceeds allowed range 1..{limit}")
    return struct.pack("!I", len(body)) + body


def decode_body(body):
    try:
        value = json.loads(body)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise FrameError("invalid RPC JSON payload") from exc
    if not isinstance(value, dict):
        raise FrameError("RPC payload must be an object")
    return value


def read_frame(stream, limit=DEFAULT_MAX_FRAME_BYTES):
    def exact(size, allow_eof=False):
        parts = []
        while size:
            chunk = stream.read(size)
            if not chunk:
                if allow_eof and not parts:
                    return None
                raise FrameError("incomplete RPC frame")
            parts.append(chunk)
            size -= len(chunk)
        return b"".join(parts)

    header = exact(HEADER_BYTES, allow_eof=True)
    if header is None:
        return None
    return decode_body(exact(frame_size(header, limit)))
