from __future__ import annotations

from typing import Protocol


class Random(Protocol):
    def randint(self, a: int, b: int) -> int: ...

    def random(self) -> float: ...
