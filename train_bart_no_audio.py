"""Compatibility wrapper for the legacy no-audio BART training path."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    target = Path(__file__).with_name("train_bart.py")
    command = [sys.executable, str(target), "--no-audio", *sys.argv[1:]]
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
