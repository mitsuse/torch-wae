from __future__ import annotations

from pathlib import Path

import torch
import typer

from torch_wae.network import WAENet, WithResample

app = typer.Typer()


@app.command()
def main(
    pt: Path = typer.Option(
        ...,
        help="the path of a model-paramter file formatted for PyTorch",
    ),
    sample_rate: int = typer.Option(
        48000,
        help="the original sample-rate for input (NOTE: WAENet resamples audio to 16KHz)",
    ),
    output: Path = typer.Option(
        ...,
        help="the output path of a model converted for ONNX",
    ),
) -> None:
    assert sample_rate >= WAENet.SAMPLE_RATE

    f = WAENet(s=1)
    f.load_state_dict(torch.load(pt))
    f.train(False)

    m = WithResample(f, sample_rate=sample_rate)

    waveform = torch.randn((1, sample_rate))

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        m,
        waveform,
        str(output),
        export_params=True,
        input_names=["waveform"],
        output_names=["z"],
        dynamic_axes={
            "waveform": {0: "batch_size"},
            "z": {0: "batch_size"},
        },
    )


if __name__ == "__main__":
    app()
