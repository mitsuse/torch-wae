from __future__ import annotations

from random import Random

import torch


def omit_silence(
    waveform: torch.Tensor,
    threshold: float,
    channel_first: bool = False,
) -> torch.Tensor:
    abs_waveform = torch.abs(waveform)
    not_silent = abs_waveform > threshold
    indices_not_silent = torch.where(
        torch.any(not_silent, dim=1 if channel_first else 0)
    )[0]

    if len(indices_not_silent) == 0:
        waveform = torch.zeros_like(waveform)
    else:
        begin_non_silence = indices_not_silent[0]
        end_non_silence = indices_not_silent[-1]
        if channel_first:
            waveform = waveform[begin_non_silence : end_non_silence + 1, :]
        else:
            waveform = waveform[:, begin_non_silence : end_non_silence + 1]

    return waveform


def add_noise(
    random: Random,
    min_noise: float,
    max_noise: float,
    waveform: torch.Tensor,
) -> torch.Tensor:
    noise_level = min_noise + random.random() * max(0.0, max_noise - min_noise)
    noise = torch.randn_like(waveform) * noise_level
    return waveform + noise


def crop_randomly(
    random: Random,
    sample_rate: int,
    durations: int,
    waveform: torch.Tensor,
) -> torch.Tensor:
    c, d = waveform.shape
    size = sample_rate * durations
    pad = max(0, size - d)

    if pad == 0:
        start = random.randint(0, d - size)
        end = start + size
        return waveform[:, start:end]
    else:
        p = torch.zeros((c, pad), dtype=waveform.dtype).to(waveform.device)
        return torch.cat((waveform, p), dim=-1)
