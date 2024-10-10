from __future__ import annotations

import torch

from torch_wae.audio import mask_randomly


class StaticRandom:
    def __init__(self, value: int) -> None:
        self.__value = value

    def randint(self, a: int, b: int) -> int:
        return self.__value

    def random(self) -> float:
        raise NotImplementedError


def test__mask_randomly() -> None:
    random = StaticRandom(value=8)
    sample_rate = 16
    w = torch.ones((2, sample_rate), dtype=torch.float)
    w = mask_randomly(random, sample_rate, 0.2, w)
    m = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1],
        ]
    )

    assert torch.all(w == m)
