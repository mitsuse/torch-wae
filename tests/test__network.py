from __future__ import annotations

import torch

from torch_wae.network import LightUNet, Preprocess, WAENet


def test__preprocess_shape() -> None:
    d = 64
    f = Preprocess().train(False)
    waveform = torch.randn((1, Preprocess.SAMPLE_RATE))
    x = f(waveform)
    assert x.shape == (1, 1, d, d)


def test__light_unet_shape() -> None:
    for s in (1, 2):
        d = 64
        f = LightUNet(s=s).train(False)
        x = torch.randn((1, 1, d, d))
        x_ = f(x)
        assert x_.shape == x.shape


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
        waveform = torch.randn((1, f.preprocess.SAMPLE_RATE))
        z = f(waveform)
        assert z.shape == (1, d * s)
