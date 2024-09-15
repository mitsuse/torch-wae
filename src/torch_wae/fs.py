from __future__ import annotations

from pathlib import Path


def basename(path: Path) -> str:
    return path.name.rsplit(".", maxsplit=1)[0]
