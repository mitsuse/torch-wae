from __future__ import annotations

import torch
from convmelspec.stft import ConvertibleSpectrogram as Spectrogram
from torch import nn
from torch.nn import functional as F


# Wowrd Audio Encoder - A network for audio similar to MobileNet V2 for images.
class WAENet(nn.Module):
    SAMPLE_RATE: int = 16000

    def __init__(self, s: int) -> None:
        super().__init__()

        melSpec = Spectrogram(
            sr=self.SAMPLE_RATE,
            n_fft=512,
            hop_size=260,
            n_mel=64,
            power=2,
        )
        melSpec.set_mode("DFT", "on_the_fly")

        self.preprocess = melSpec

        self.encoder = Encoder(s=s)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        feature = self.preprocess(waveform)
        feature = torch.clip(feature, 1e-6)
        feature = torch.log(feature)
        feature = feature[:, None, :, :]
        z = self.encoder(feature)
        return z


class Encoder(nn.Module):
    def __init__(self, s: int) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            # --------------------
            # shape: (1, 64, 64) -> (16, 32, 32)
            # --------------------
            InvertedBottleneck(k=3, c_in=1, c_out=16 * s, stride=2),
            # --------------------
            # shape: (16, 32, 32) -> (8, 32, 32)
            # --------------------
            InvertedBottleneck(k=3, c_in=16 * s, c_out=8 * s, stride=1),
            InvertedBottleneck(k=3, c_in=8 * s, c_out=8 * s, stride=1),
            # --------------------
            # shape: (8, 32, 32) -> (12, 16, 16)
            # --------------------
            InvertedBottleneck(k=3, c_in=8 * s, c_out=12 * s, stride=2),
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
            # --------------------
            # shape: (64, 4, 4) -> (64, 1, 1)
            # --------------------
            nn.Conv2d(64 * s, 64 * s, 4, stride=1),
            nn.BatchNorm2d(64 * s),
            nn.LeakyReLU(),
            nn.Conv2d(64 * s, 64 * s, 1, stride=1),
            # --------------------
            # normalize
            # --------------------
            nn.Flatten(),
            L2Normalize(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.layers(x)


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
