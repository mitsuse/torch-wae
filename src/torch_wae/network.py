from __future__ import annotations

import torch
from convmelspec.stft import ConvertibleSpectrogram as Spectrogram
from torch import nn
from torch.nn import functional as F
from torchaudio import functional as FA


# Wowrd Audio Encoder - A network for audio similar to MobileNet V2 for images.
class WAENet(nn.Module):
    def __init__(self, s: int) -> None:
        super().__init__()

        self.preprocess = Preprocess()

        self.encoder = Encoder(s=s)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        x = self.preprocess(waveform)
        z = self.encoder(x)
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
