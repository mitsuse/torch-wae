from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torchaudio
import typer
from torchaudio import functional as FA

app = typer.Typer()


@app.command()
def main(
    n_workers: int = typer.Option(
        ...,
        help="the number of workers used to transform dataset",
    ),
    root: Path = typer.Option(
        ...,
        help="the root path of a directory which contains datasets.",
    ),
    annotation: Path = typer.Option(
        ...,
        help="the path of a dataset to be encoded.",
    ),
    max_examples: int = typer.Option(
        ...,
        help="the max number of examples for a shard.",
    ),
    output: Path = typer.Option(
        ...,
        help="the output path of a directory which stores shards of WebDataset,",
    ),
) -> None:
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from webdataset.writer import ShardWriter

    from torch_wae.dataset import JsonLinesDataset, transformed_iter

    dataset = JsonLinesDataset(annotation)

    n = 0
    for _ in dataset:
        n += 1

    loader = DataLoader(
        transformed_iter(
            dataset,
            Transform(
                root=root,
                resample_rate=16000,
                durations=1,
            ),
        ),
        batch_size=1,
        shuffle=False,
        num_workers=n_workers,
    )

    output.mkdir(parents=True, exist_ok=True)
    pattern = str(output / "%06d.tar")

    with tqdm(total=n) as progress:
        with ShardWriter(pattern, maxcount=max_examples, verbose=0) as w:
            for i, (path, key, class_id, waveform, ignore) in enumerate(loader):
                if n <= i:
                    break

                ignore = bool(ignore.detach().numpy()[0])
                if ignore:
                    continue

                path = path[0]
                key = key[0]
                class_id = int(class_id.detach().numpy()[0])
                waveform = waveform.detach().numpy()[0]

                w.write(
                    {
                        "__key__": key,
                        "npy": waveform,
                        "json": {
                            "class_id": class_id,
                            "path": path,
                        },
                    }
                )

                progress.update(1)


class Transform:
    def __init__(
        self,
        root: Path,
        resample_rate: int,
        durations: int,
    ) -> None:
        super().__init__()

        self.__root = root
        self.__resample_rate = resample_rate
        self.__durations = durations

    def __call__(self, example: Any) -> tuple[str, str, int, torch.Tensor, bool]:
        from torch_wae.audio import crop_or_pad_last

        root = self.__root
        resample_rate = self.__resample_rate
        durations = self.__durations

        path = str(example["path"])
        key = path.rsplit(".", maxsplit=1)[0]
        class_id = int(example["class_id"])

        path_file = root / path

        waveform, sample_rate = torchaudio.load(path_file)
        waveform = torch.mean(waveform, dim=0).unsqueeze(0)

        if resample_rate != sample_rate:
            waveform = FA.resample(waveform, sample_rate, resample_rate)

        frames = waveform.shape[-1]

        waveform = crop_or_pad_last(resample_rate, durations, waveform)

        ignore = frames > durations * resample_rate

        return path, key, class_id, waveform, ignore


if __name__ == "__main__":
    app()
