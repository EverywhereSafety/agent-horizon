"""Stateless model-code worker; receives no environment dynamics or database."""

import contextlib
import io
import json
import resource
import sys
import traceback
import linecache
import re
from pathlib import PurePath

resource.setrlimit(resource.RLIMIT_FSIZE, (32 * 1024**2, 32 * 1024**2))
resource.setrlimit(resource.RLIMIT_NOFILE, (128, 128))
request = json.loads(sys.stdin.readline())
memory_bytes = request.get("memory_bytes", 8 * 1024**3)
if type(memory_bytes) is not int or memory_bytes <= 0:
    raise ValueError("invalid Python memory budget")
resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
namespace = {"__name__": "__main__"}
sys.path.insert(0, "/workspace")
import math

namespace["math"] = math
namespace["json"] = json
try:
    import numpy

    namespace["np"] = numpy
    namespace["numpy"] = numpy
    import scipy

    namespace["scipy"] = scipy
except ImportError:
    pass
filename = "<model-tool-code>"
code = ""
reading_source = False
lines = []
buf = io.StringIO()
errbuf = io.StringIO()
try:
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(errbuf):
        if request.get("path") is not None:
            from pathlib import Path

            source = Path("/workspace") / request["path"]
            if not source.resolve().is_relative_to(Path("/workspace")):
                raise ValueError("path outside workspace")
            filename = str(source)
            reading_source = True
            code = source.read_text()
            reading_source = False
            namespace["__file__"] = filename
        else:
            code = request["code"]
        lines = code.splitlines(keepends=True)
        linecache.cache[filename] = (len(code), None, lines, filename)
        exec(compile(code, filename, "exec"), namespace)
    result = {"output": buf.getvalue()[:16384] or "(no stdout)", "done": False}
except SystemExit as exc:
    result = {"output": buf.getvalue()[:16384] or "(no stdout)", "done": False}
    if exc.code not in (None, 0):
        result["error"] = f"SystemExit: {exc.code}"
except Exception as exc:
    if (
        reading_source
        and isinstance(exc, OSError)
        and not isinstance(
            exc, (FileNotFoundError, IsADirectoryError, NotADirectoryError)
        )
    ):
        raise
    formatted = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    # Keep model source and package names, without host filesystem locations.
    formatted = re.sub(
        r'File "([^"]+)"',
        lambda m: 'File "'
        + (m[1] if m[1] == "<model-tool-code>" else PurePath(m[1]).name)
        + '"',
        formatted,
    )
    lineno = exc.lineno if isinstance(exc, SyntaxError) else None
    if lineno is None:
        frames = traceback.extract_tb(exc.__traceback__)
        lineno = next(
            (f.lineno for f in reversed(frames) if f.filename == filename), None
        )
    excerpt = ""
    if lineno is not None:
        excerpt = "\n".join(
            f"{i+1}: {lines[i].rstrip()}"
            for i in range(max(0, lineno - 3), min(len(lines), lineno + 2))
        )
    result = {
        "error": f"{type(exc).__name__}: {exc}"[:1024],
        "traceback": formatted[-8192:],
        "traceback_truncated": len(formatted) > 8192,
        "line_number": lineno,
        "code_excerpt": excerpt[:2048],
        "output": buf.getvalue()[:2048],
        "done": False,
    }
result["stderr"] = errbuf.getvalue()[:16384]
print(json.dumps(result), flush=True)
