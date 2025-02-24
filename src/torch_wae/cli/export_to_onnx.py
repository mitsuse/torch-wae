from __future__ import annotations

from pathlib import Path

import torch
import typer
from pydantic.dataclasses import dataclass

from torch_wae.network import (
    Preprocess,
    WAEActivationType,
    WAEHeadType,
    WAENet,
    WithResample,
)

app = typer.Typer()


@dataclass(frozen=True)
class Config:
    wae: WAEConfig


@dataclass(frozen=True)
class WAEConfig:
    activation: WAEActivationType
    head: WAEHeadType
    s: int


@app.command()
def main(
    pt: Path = typer.Option(
        ...,
        help="the path of a model-paramter file formatted for PyTorch",
    ),
    config: Path = typer.Option(
        ...,
        help="the path of a config file used for training model",
    ),
    shift_melspec: float = typer.Option(
        ...,
        help="a hyper-paramerter to shift log mel-spectrogram",
    ),
    output: Path = typer.Option(
        ...,
        help="the output path of a model converted for ONNX",
    ),
) -> None:
    import yaml

    with config.open() as f:
        c = Config(**yaml.load(f, Loader=yaml.CLoader))

    f = WAENet(
        activation_type=c.wae.activation,
        head_type=c.wae.head,
        s=c.wae.s,
        shift_melspec=shift_melspec,
    )

    f.load_state_dict(torch.load(pt))
    f.train(False)

    m = WithResample(f, Preprocess.SAMPLE_RATE)

    waveform = torch.randn((1, 48000))

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        m,
        (waveform,),
        str(output),
        export_params=True,
        input_names=["waveform"],
        output_names=["z"],
        dynamic_axes={
            "waveform": {0: "batch_size", 1: "sample_rate"},
            "z": {0: "batch_size"},
        },
    )


if __name__ == "__main__":
    app()
