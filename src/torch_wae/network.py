from __future__ import annotations

import logging
from enum import Enum

import torch
from convmelspec.stft import ConvertibleSpectrogram as Spectrogram
from torch import nn
from torch.nn import functional as F
from torchaudio import functional as FA


# Word Audio Encoder - A network for audio similar to MobileNet V2 for images.
class WAEHeadType(str, Enum):
    CONV_H1 = "conv_H1"
    CONV_H2 = "conv_H2"
    CONV_H4 = "conv_H4"
    LINEAR = "linear"
    ATTEN_1 = "atten_1"
    ATTEN_2 = "atten_2"
    ATTEN_4 = "atten_4"


class WAEActivationType(str, Enum):
    LEAKY_RELU = "leaky_relu"
    RELU6 = "relu6"
    TANH = "tanh"


def activation(type_: WAEActivationType) -> nn.Module:
    match type_:
        case WAEActivationType.RELU6:
            activation: nn.Module = nn.ReLU6()
        case WAEActivationType.LEAKY_RELU:
            activation = nn.LeakyReLU()
        case WAEActivationType.TANH:
            activation = nn.Tanh()
        case _:
            raise ValueError(f"unknown activation type: {type_}")

    return activation


class WAENet(nn.Module):
    def __init__(
        self,
        s: int,
        head_type: WAEHeadType,
        activation_type: WAEActivationType,
        head_activation_type: WAEActivationType,
    ) -> None:
        super().__init__()

        self.preprocess = Preprocess()

        self.encoder = Encoder(s=s, activation_type=activation_type)

        match head_type:
            case WAEHeadType.CONV_H1:
                self.head: nn.Module = WAEConvHead(
                    activation_type=head_activation_type,
                    s=s,
                    h=1,
                )
            case WAEHeadType.CONV_H2:
                self.head = WAEConvHead(
                    activation_type=head_activation_type,
                    s=s,
                    h=2,
                )
            case WAEHeadType.CONV_H4:
                self.head = WAEConvHead(
                    activation_type=head_activation_type,
                    s=s,
                    h=4,
                )
            case WAEHeadType.LINEAR:
                self.head = WAELinearHead(
                    activation_type=head_activation_type,
                    s=s,
                )
            case WAEHeadType.ATTEN_1:
                self.head = WAEAttentionHead(
                    n_head=1,
                    s=s,
                )
            case WAEHeadType.ATTEN_2:
                self.head = WAEAttentionHead(
                    n_head=2,
                    s=s,
                )
            case WAEHeadType.ATTEN_4:
                self.head = WAEAttentionHead(
                    n_head=4,
                    s=s,
                )
            case _:
                raise ValueError(f"unknown head type: {head_type}")

        self.norm = L2Normalize()

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        h = self.preprocess(waveform)
        h = self.encoder(h)
        h = self.head(h)
        z = self.norm(h)
        return z


class Encoder(nn.Module):
    def __init__(self, s: int, activation_type: WAEActivationType) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            # --------------------
            # shape: (1, 64, 64) -> (8, 32, 32)
            # --------------------
            InvertedBottleneck(
                k=3,
                c_in=1,
                c_out=8 * s,
                stride=2,
                activation_type=activation_type,
            ),
            # --------------------
            # shape: (8, 32, 32) -> (12, 32, 32)
            # --------------------
            InvertedBottleneck(
                k=3,
                c_in=8 * s,
                c_out=12 * s,
                stride=1,
                activation_type=activation_type,
            ),
            InvertedBottleneck(
                k=3,
                c_in=12 * s,
                c_out=12 * s,
                stride=1,
                activation_type=activation_type,
            ),
            # --------------------
            # shape: (12, 32, 32) -> (12, 16, 16)
            # --------------------
            InvertedBottleneck(
                k=3,
                c_in=12 * s,
                c_out=12 * s,
                stride=2,
                activation_type=activation_type,
            ),
            InvertedBottleneck(
                k=3,
                c_in=12 * s,
                c_out=12 * s,
                stride=1,
                activation_type=activation_type,
            ),
            InvertedBottleneck(
                k=3,
                c_in=12 * s,
                c_out=12 * s,
                stride=1,
                activation_type=activation_type,
            ),
            # --------------------
            # shape: (12, 16, 16) -> (16, 8, 8)
            # --------------------
            InvertedBottleneck(
                k=3,
                c_in=12 * s,
                c_out=16 * s,
                stride=2,
                activation_type=activation_type,
            ),
            InvertedBottleneck(
                k=3,
                c_in=16 * s,
                c_out=16 * s,
                stride=1,
                activation_type=activation_type,
            ),
            InvertedBottleneck(
                k=3,
                c_in=16 * s,
                c_out=16 * s,
                stride=1,
                activation_type=activation_type,
            ),
            InvertedBottleneck(
                k=3,
                c_in=16 * s,
                c_out=16 * s,
                stride=1,
                activation_type=activation_type,
            ),
            # --------------------
            # shape: (16, 8, 8) -> (32, 4, 4)
            # --------------------
            InvertedBottleneck(
                k=3,
                c_in=16 * s,
                c_out=32 * s,
                stride=2,
                activation_type=activation_type,
            ),
            InvertedBottleneck(
                k=3,
                c_in=32 * s,
                c_out=32 * s,
                stride=1,
                activation_type=activation_type,
            ),
            InvertedBottleneck(
                k=3,
                c_in=32 * s,
                c_out=32 * s,
                stride=1,
                activation_type=activation_type,
            ),
            # --------------------
            # shape: (32, 4, 4) -> (64, 4, 4)
            # --------------------
            InvertedBottleneck(
                k=3,
                c_in=32 * s,
                c_out=64 * s,
                stride=1,
                activation_type=activation_type,
            ),
            InvertedBottleneck(
                k=3,
                c_in=64 * s,
                c_out=64 * s,
                stride=1,
                activation_type=activation_type,
            ),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        return self.layers(x)


class WAEConvHead(nn.Module):
    def __init__(
        self,
        activation_type: WAEActivationType,
        s: int,
        h: int,
    ) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            # --------------------
            # shape: (64, 4, 4) -> (64, 1, 1)
            # --------------------
            nn.Conv2d(64 * s, 64 * s * h, 4, stride=1),
            nn.BatchNorm2d(64 * s * h),
            activation(activation_type),
            nn.Conv2d(64 * s * h, 64 * s, 1, stride=1),
            nn.Flatten(),
        )

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor]:
        return self.layers(h)


class WAELinearHead(nn.Module):
    def __init__(
        self,
        activation_type: WAEActivationType,
        s: int,
    ) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            # --------------------
            # shape: (64, 4, 4) -> (1024,)
            # --------------------
            nn.Flatten(),
            # --------------------
            # shape: (1024,) -> (64,)
            # --------------------
            nn.BatchNorm1d(1024 * s),
            activation(activation_type),
            nn.Linear(1024 * s, 64 * s),
        )

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor]:
        return self.layers(h)


class WAEAttentionHead(nn.Module):
    def __init__(
        self,
        n_head: int,
        s: int,
    ) -> None:
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=64 * s,
            num_heads=n_head,
            batch_first=True,
        )

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor]:
        batch_size, d_s, height, width = h.shape
        h = h.permute(0, 2, 3, 1).reshape(batch_size, height * width, d_s)
        h, _ = self.attention(h, h, h)
        z = h.mean(dim=1)  # shape: (batch_size, d * s)
        return z


class Preprocess(nn.Module):
    SAMPLE_RATE: int = 16000

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()

        self.melSpec = Spectrogram(
            sr=self.SAMPLE_RATE,
            n_fft=512,
            hop_size=260,
            n_mel=64,
            power=2,
        )
        self.melSpec.set_mode("DFT", "on_the_fly")
        self.eps = eps

        with torch.no_grad():
            x = torch.zeros((1, self.SAMPLE_RATE))
            x = self.melSpec(x)
            self.log_mel_min = torch.log(x + self.eps).min()

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        x = self.melSpec(waveform)
        x = torch.clip(torch.log(x + self.eps) - self.log_mel_min, 0.0)
        x = x.unsqueeze(1)
        return x


class InvertedBottleneck(nn.Module):
    def __init__(
        self,
        k: int,
        c_in: int,
        c_out: int,
        stride: int,
        activation_type: WAEActivationType,
    ) -> None:
        super().__init__()

        c_hidden = c_in * 3
        padding = int((k - 1) / 2)

        self.main = nn.Sequential(
            nn.Conv2d(c_in, c_hidden, 1, stride=1),
            nn.BatchNorm2d(c_hidden),
            activation(activation_type),
            nn.Conv2d(
                c_hidden,
                c_hidden,
                k,
                groups=c_hidden,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm2d(c_hidden),
            activation(activation_type),
            nn.Conv2d(c_hidden, c_out, 1, stride=1),
            nn.BatchNorm2d(c_out),
        )

        self.branch = nn.Sequential(
            nn.Conv2d(c_in, c_out, 1, stride=stride),
            nn.BatchNorm2d(c_out),
            activation(activation_type),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.main(x)
        d = x if self.branch is None else self.branch(x)
        return F.leaky_relu(h + d)


class L2Normalize(nn.Module):
    def __init__(self, dim: int = 1, eps: float = 1e-12):
        super().__init__()

        self.dim = dim
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, p=2, dim=self.dim, eps=self.eps)


class WithResample(torch.nn.Module):
    def __init__(self, f: WAENet, sample_rate: int) -> None:
        super().__init__()

        self.f = f
        self.sample_rate = sample_rate

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        h = FA.resample(waveform, self.sample_rate, self.f.SAMPLE_RATE)
        z = self.f(h)
        return z
