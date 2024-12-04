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
    CONV = "conv"
    LINEAR = "linear"
    ATTEN_1 = "atten_1"
    ATTEN_2 = "atten_2"
    ATTEN_4 = "atten_4"


class WAEActivationType(str, Enum):
    LEAKY_RELU = "leaky_relu"
    TANH = "tanh"


class WAENet(nn.Module):
    def __init__(
        self,
        s: int,
        head_type: WAEHeadType,
        head_activation_type: WAEActivationType,
    ) -> None:
        super().__init__()

        self.preprocess = Preprocess()

        self.encoder = Encoder(s=s)

        if head_type not in (WAEHeadType.CONV, WAEHeadType.LINEAR):
            logging.debug(
                "`head_activation_type` is not supported with specified `head_type`"
            )

        match head_type:
            case WAEHeadType.CONV:
                self.head: nn.Module = WAEConvHead(
                    activation_type=head_activation_type,
                    s=s,
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
    def __init__(self, s: int) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            # --------------------
            # shape: (1, 64, 64) -> (8, 32, 32)
            # --------------------
            InvertedBottleneck(k=3, c_in=1, c_out=8 * s, stride=2),
            # --------------------
            # shape: (8, 32, 32) -> (12, 32, 32)
            # --------------------
            InvertedBottleneck(k=3, c_in=8 * s, c_out=12 * s, stride=1),
            InvertedBottleneck(k=3, c_in=12 * s, c_out=12 * s, stride=1),
            # --------------------
            # shape: (12, 32, 32) -> (12, 16, 16)
            # --------------------
            InvertedBottleneck(k=3, c_in=12 * s, c_out=12 * s, stride=2),
            InvertedBottleneck(k=3, c_in=12 * s, c_out=12 * s, stride=1),
            InvertedBottleneck(k=3, c_in=12 * s, c_out=12 * s, stride=1),
            # --------------------
            # shape: (12, 16, 16) -> (16, 8, 8)
            # --------------------
            InvertedBottleneck(k=3, c_in=12 * s, c_out=16 * s, stride=2),
            InvertedBottleneck(k=3, c_in=16 * s, c_out=16 * s, stride=1),
            InvertedBottleneck(k=3, c_in=16 * s, c_out=16 * s, stride=1),
            InvertedBottleneck(k=3, c_in=16 * s, c_out=16 * s, stride=1),
            # --------------------
            # shape: (16, 8, 8) -> (32, 4, 4)
            # --------------------
            InvertedBottleneck(k=3, c_in=16 * s, c_out=32 * s, stride=2),
            InvertedBottleneck(k=3, c_in=32 * s, c_out=32 * s, stride=1),
            InvertedBottleneck(k=3, c_in=32 * s, c_out=32 * s, stride=1),
            # --------------------
            # shape: (32, 4, 4) -> (64, 4, 4)
            # --------------------
            InvertedBottleneck(k=3, c_in=32 * s, c_out=64 * s, stride=1),
            InvertedBottleneck(k=3, c_in=64 * s, c_out=64 * s, stride=1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        return self.layers(x)


class WAEConvHead(nn.Module):
    def __init__(
        self,
        activation_type: WAEActivationType,
        s: int,
    ) -> None:
        super().__init__()

        match activation_type:
            case WAEActivationType.LEAKY_RELU:
                activation: nn.Module = nn.LeakyReLU()
            case WAEActivationType.TANH:
                activation = nn.Tanh()
            case _:
                raise ValueError(f"unknown activation type: {activation_type}")

        self.layers = nn.Sequential(
            # --------------------
            # shape: (64, 4, 4) -> (64, 1, 1)
            # --------------------
            nn.Conv2d(64 * s, 64 * s, 4, stride=1),
            nn.BatchNorm2d(64 * s),
            activation,
            nn.Conv2d(64 * s, 64 * s, 1, stride=1),
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

        match activation_type:
            case WAEActivationType.LEAKY_RELU:
                activation: nn.Module = nn.LeakyReLU()
            case WAEActivationType.TANH:
                activation = nn.Tanh()
            case _:
                raise ValueError(f"unknown activation type: {activation_type}")

        self.layers = nn.Sequential(
            # --------------------
            # shape: (64, 4, 4) -> (1024,)
            # --------------------
            nn.Flatten(),
            # --------------------
            # shape: (1024,) -> (64,)
            # --------------------
            nn.Linear(1024 * s, 256 * s),
            nn.BatchNorm1d(256 * s),
            activation,
            nn.Linear(256 * s, 64 * s),
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


class LightUNet(nn.Module):
    def __init__(self, s: int) -> None:
        super().__init__()

        mode = "nearest"

        # --------------------
        # shape: (1, 64, 64) -> (8, 32, 32)
        # --------------------
        self.encode_0 = nn.Sequential(
            InvertedBottleneck(k=3, c_in=1, c_out=8 * s, stride=2),
            InvertedBottleneck(k=3, c_in=8 * s, c_out=8 * s, stride=1),
            InvertedBottleneck(k=3, c_in=8 * s, c_out=8 * s, stride=1),
        )

        # --------------------
        # shape: (8, 32, 32) -> (12, 16, 16)
        # --------------------
        self.encode_1 = nn.Sequential(
            InvertedBottleneck(k=3, c_in=8 * s, c_out=12 * s, stride=2),
            InvertedBottleneck(k=3, c_in=12 * s, c_out=12 * s, stride=1),
            InvertedBottleneck(k=3, c_in=12 * s, c_out=12 * s, stride=1),
        )

        # --------------------
        # shape: (12, 16, 16) -> (16, 8, 8)
        # --------------------
        self.encode_2 = nn.Sequential(
            InvertedBottleneck(k=3, c_in=12 * s, c_out=16 * s, stride=2),
            InvertedBottleneck(k=3, c_in=16 * s, c_out=16 * s, stride=1),
            InvertedBottleneck(k=3, c_in=16 * s, c_out=16 * s, stride=1),
            InvertedBottleneck(k=3, c_in=16 * s, c_out=16 * s, stride=1),
        )

        # --------------------
        # shape: (16, 8, 8) -> (32, 4, 4)
        # --------------------
        self.encode_3 = nn.Sequential(
            InvertedBottleneck(k=3, c_in=16 * s, c_out=32 * s, stride=2),
            InvertedBottleneck(k=3, c_in=32 * s, c_out=32 * s, stride=1),
            InvertedBottleneck(k=3, c_in=32 * s, c_out=32 * s, stride=1),
        )

        # --------------------
        # shape: (32, 4, 4) -> (16, 8, 8)
        # --------------------
        self.decode_0 = nn.Sequential(
            InvertedBottleneck(k=3, c_in=32 * s, c_out=32 * s, stride=1),
            InvertedBottleneck(k=3, c_in=32 * s, c_out=32 * s, stride=1),
            nn.Upsample(scale_factor=2, mode=mode),
            InvertedBottleneck(k=3, c_in=32 * s, c_out=16 * s, stride=1),
        )

        # --------------------
        # shape: (16, 8, 8) -> (12, 16, 16)
        # --------------------
        self.decode_1 = nn.Sequential(
            InvertedBottleneck(k=3, c_in=16 * s, c_out=16 * s, stride=1),
            InvertedBottleneck(k=3, c_in=16 * s, c_out=16 * s, stride=1),
            InvertedBottleneck(k=3, c_in=16 * s, c_out=16 * s, stride=1),
            nn.Upsample(scale_factor=2, mode=mode),
            InvertedBottleneck(k=3, c_in=16 * s, c_out=12 * s, stride=1),
        )

        # --------------------
        # shape: (12, 16, 16) -> (8, 32, 32)
        # --------------------
        self.decode_2 = nn.Sequential(
            InvertedBottleneck(k=3, c_in=12 * s, c_out=12 * s, stride=1),
            InvertedBottleneck(k=3, c_in=12 * s, c_out=12 * s, stride=1),
            InvertedBottleneck(k=3, c_in=12 * s, c_out=12 * s, stride=1),
            InvertedBottleneck(k=3, c_in=12 * s, c_out=12 * s, stride=1),
            nn.Upsample(scale_factor=2, mode=mode),
            InvertedBottleneck(k=3, c_in=12 * s, c_out=8 * s, stride=1),
        )

        # --------------------
        # shape: (8, 32, 32) -> (1, 64, 64)
        # --------------------
        self.decode_3 = nn.Sequential(
            InvertedBottleneck(k=3, c_in=8 * s, c_out=1 * s, stride=1),
            nn.Upsample(scale_factor=2, mode=mode),
            InvertedBottleneck(k=3, c_in=1 * s, c_out=1 * s, stride=1),
        )

        # --------------------
        # Refine
        # --------------------
        self.refine = nn.Sequential(
            nn.Conv2d(1 * s, 1 * s, 3, stride=1, padding=1),
            nn.BatchNorm2d(1 * s),
            nn.LeakyReLU(),
            nn.Conv2d(1 * s, 1, 3, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h_0 = self.encode_0(x)
        h_1 = self.encode_1(h_0)
        h_2 = self.encode_2(h_1)
        h_3 = self.encode_3(h_2)

        h_4 = self.decode_0(h_3)
        h_5 = self.decode_1(h_4 + h_2)
        h_6 = self.decode_2(h_5 + h_1)
        h_7 = self.decode_3(h_6 + h_0)

        return self.refine(h_7 + x)


class Preprocess(nn.Module):
    SAMPLE_RATE: int = 16000

    def __init__(self) -> None:
        super().__init__()

        self.melSpec = Spectrogram(
            sr=self.SAMPLE_RATE,
            n_fft=512,
            hop_size=260,
            n_mel=64,
            power=2,
        )
        self.melSpec.set_mode("DFT", "on_the_fly")

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        x = self.melSpec(waveform)
        x = torch.clip(x, 1e-6)
        x = torch.log(x)
        x = x[:, None, :, :]
        return x


class InvertedBottleneck(nn.Module):
    def __init__(self, k: int, c_in: int, c_out: int, stride: int) -> None:
        super().__init__()

        c_hidden = c_in * 3
        padding = int((k - 1) / 2)

        self.main = nn.Sequential(
            nn.Conv2d(c_in, c_hidden, 1, stride=1),
            nn.BatchNorm2d(c_hidden),
            nn.LeakyReLU(),
            nn.Conv2d(
                c_hidden,
                c_hidden,
                k,
                groups=c_hidden,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm2d(c_hidden),
            nn.LeakyReLU(),
            nn.Conv2d(c_hidden, c_out, 1, stride=1),
            nn.BatchNorm2d(c_out),
        )

        self.branch = nn.Sequential(
            nn.Conv2d(c_in, c_out, 1, stride=stride),
            nn.BatchNorm2d(c_out),
            nn.LeakyReLU(),
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
