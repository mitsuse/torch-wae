from __future__ import annotations

from pathlib import Path
from typing import Any

import typer

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
    size_shard: int = typer.Option(
        20,
        help="the max size of a shard (unit: MB)",
    ),
    output: Path = typer.Option(
        ...,
        help="the output path of a directory which stores shards of WebDataset.",
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
            Transform(),
        ),
        batch_size=1,
        shuffle=False,
        num_workers=n_workers,
    )

    output.mkdir(parents=True, exist_ok=True)
    pattern = str(output / "%04d.tar")
    max_size = size_shard * 1024**2

    with tqdm(total=n) as progress:
        with ShardWriter(pattern, maxsize=max_size, verbose=0) as w:
            for i, (anchor, positive, key, class_id) in enumerate(loader):
                if n <= i:
                    break

                anchor = anchor[0]
                positive = positive[0]
                key = key[0]
                class_id = int(class_id.detach().numpy()[0])

                w.write(
                    {
                        "__key__": key,
                        "json": {
                            "class_id": class_id,
                            "anchor": anchor,
                            "positive": positive,
                        },
                    }
                )

                progress.update(1)


class Transform:
    def __init__(self) -> None: ...

    def __call__(self, example: Any) -> tuple[str, str, str, int]:
        from torch_wae import fs

        anchor = Path(example["anchor"])
        positive = Path(example["positive"])

        path = anchor.parent
        basename_anchor = fs.basename(anchor)
        basename_positive = fs.basename(positive)

        path_anchor = anchor.parent / f"{basename_anchor}.npy"
        path_positive = positive.parent / f"{basename_positive}.npy"

        key = str(path / f"{basename_anchor}-{basename_positive}")
        class_id = int(example["class_id"])

        return str(path_anchor), str(path_positive), key, class_id


if __name__ == "__main__":
    app()
