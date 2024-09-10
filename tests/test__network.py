from __future__ import annotations

import torch

from torch_wae.network import WAENet


def test__wae_preprocess_shape() -> None:
    s = 1
    d = 64
    f = WAENet(s=s).train(False)
    waveform = torch.randn((1, WAENet.SAMPLE_RATE))
    features = f.preprocess(waveform)
    assert features.shape == (1, d, d)


def test__wae_encoder_shape() -> None:
    for s in (1, 2):
        d = 64
        f = WAENet(s=s).train(False)
        x = torch.randn((1, 1, d, d))
        z = f.encoder(x)
        assert z.shape == (1, d * s)


def test__wae_forward_shape() -> None:
    for s in (1, 2):
        d = 64
        f = WAENet(s=s).train(False)
        waveform = torch.randn((1, WAENet.SAMPLE_RATE))
        z = f(waveform)
        assert z.shape == (1, d * s)
