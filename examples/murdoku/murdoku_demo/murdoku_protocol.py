"""Training bridge to Murdoku Lab's canonical task protocol."""

from murdoku_lab.environment.protocol import (
    PROTOCOL,
    SCHEMAS,
    WORKSPACE_SCHEMA,
    simplify_record,
    parse_function_call,
)

__all__ = [
    "PROTOCOL",
    "SCHEMAS",
    "WORKSPACE_SCHEMA",
    "simplify_record",
    "parse_function_call",
]
