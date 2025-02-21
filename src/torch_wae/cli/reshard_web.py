from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import typer

app = typer.Typer()


@app.command()
def main(
    n_workers: int = typer.Option(
        ...,
        help="the number of workers used to transform dataset",
    ),
    dataset: Path = typer.Option(
        ...,
        help="the path of a directory which stores original shards",
    ),
    approximation: int = typer.Option(
        ...,
        help="the approximate size of dataset",
    ),
    max_count: int = typer.Option(
        5000,
        help="the max count of examples in a shard",
    ),
    output: Path = typer.Option(
        ...,
        help="the output path of a directory which stores shards of WebDataset,",
    ),
) -> None:
    import os

    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from webdataset import WebDataset
    from webdataset.writer import ShardWriter

    seq_dataset = os.listdir(dataset)
    seq_dataset = filter(lambda s: s.endswith(".tar.gz"), seq_dataset)
    seq_dataset = map(lambda s: str(dataset / s), seq_dataset)
    seq_dataset = list(seq_dataset)

    loader = DataLoader(
        WebDataset(seq_dataset, shardshuffle=False).map(Transform()),
        batch_size=None,
        shuffle=False,
        num_workers=n_workers,
    )

    output.mkdir(parents=True, exist_ok=True)
    pattern = str(output / "%06d.tar.gz")

    with tqdm(total=approximation) as progress:
        with ShardWriter(pattern, maxcount=max_count, verbose=0) as w:
            for example in loader:
                example["npy"] = example["npy"].detach().numpy()
                w.write(example)
                progress.update(1)


class Transform:
    def __call__(self, example: Any) -> Any:
        import json
        from io import BytesIO

        key = str(example["__key__"])
        json_ = json.loads(example["json"])
        melspec = np.lib.format.read_array(BytesIO(example["npy"]))  # type: ignore

        return {
            "__key__": key,
            "npy": melspec,
            "json": json_,
        }


if __name__ == "__main__":
    app()
