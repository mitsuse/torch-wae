from __future__ import annotations

import torch

from torch_wae.network import (
    Encoder,
    Preprocess,
    WAEActivationType,
    WAEAttentionHead,
    WAEConvHead,
    WAEHeadType,
    WAELinearHead,
    WAENet,
)


def test__preprocess_shape() -> None:
    d = 64
    f = Preprocess().train(False)
    waveform = torch.randn((1, Preprocess.SAMPLE_RATE))
    x = f(waveform)
    assert x.shape == (1, 1, d, d)


def test__wae_encoder_shape() -> None:
    for s in (1, 2):
        d = 64
        f = Encoder(s=s, activation_type=WAEActivationType.RELU6).train(False)
        x = torch.randn((1, 1, d, d))
        z = f(x)
        assert z.shape == (1, d * s, 4, 4)


def test__wae_conv_head_shape() -> None:
    for s in (1, 2):
        d = 64
        f = WAEConvHead(
            activation_type=WAEActivationType.LEAKY_RELU,
            s=s,
        ).train(False)
        x = torch.randn((1, d * s, 4, 4))
        z = f(x)
        assert z.shape == (1, d * s)


def test__wae_linear_head_shape() -> None:
    for s in (1, 2):
        d = 64
        f = WAELinearHead(
            activation_type=WAEActivationType.LEAKY_RELU,
            s=s,
        ).train(False)
        x = torch.randn((1, d * s, 4, 4))
        z = f(x)
        assert z.shape == (1, d * s)


def test__wae_attention_head_shape() -> None:
    for s in (1, 2):
        d = 64
        f = WAEAttentionHead(
            n_head=2,
            s=s,
        ).train(False)
        x = torch.randn((1, d * s, 4, 4))
        z = f(x)
        assert z.shape == (1, d * s)


def test__wae_forward_shape() -> None:
    seq_head = (
        WAEHeadType.CONV,
        WAEHeadType.LINEAR,
        WAEHeadType.ATTEN_1,
        WAEHeadType.ATTEN_2,
    )
    seq_s = (1, 2)

    for head_type in seq_head:
        for s in seq_s:
            d = 64
            f = WAENet(
                head_type=head_type,
                activation_type=WAEActivationType.RELU6,
                head_activation_type=WAEActivationType.RELU6,
                s=s,
            ).train(False)
            waveform = torch.randn((1, f.preprocess.SAMPLE_RATE))
            z = f(waveform)
            assert z.shape == (1, d * s)
