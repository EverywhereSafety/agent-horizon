"""Training demo entry point for Murdoku Lab's native text/vision benchmark."""

import asyncio
import os
import sys

from murdoku_lab.evaluation.tools import main, p, tool_package_requirements

if __name__ == "__main__":
    asyncio.run(main(p.parse_args()))
