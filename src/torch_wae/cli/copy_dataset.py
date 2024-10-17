from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer()


@app.command()
def main(
    dataset: Path = typer.Option(..., help="the path of a dataset to be copied."),
    output: Path = typer.Option(..., help=""),
) -> None:
    import json
    import shutil

    from tqdm import tqdm

    root_dataset = dataset.parent

    with dataset.open() as f:
        n = sum(1 for _ in f)

    with tqdm(total=n) as progress, dataset.open() as f:
        for line in f:
            json_example = json.loads(line)

            path_rel_anchor = str(json_example["anchor"])
            path_rel_positive = str(json_example["positive"])

            path_anchor_src = root_dataset / path_rel_anchor
            path_positive_src = root_dataset / path_rel_positive

            path_anchor_dst = output / path_rel_anchor
            path_positive_dst = output / path_rel_positive

            path_anchor_dst.parent.mkdir(parents=True, exist_ok=True)
            path_positive_dst.parent.mkdir(parents=True, exist_ok=True)

            shutil.copy(path_anchor_src, path_anchor_dst)
            shutil.copy(path_positive_src, path_positive_dst)

            progress.update(1)

    output.mkdir(parents=True, exist_ok=True)

    dataset_copied = output / dataset.name
    dataset_copied.write_bytes(dataset.read_bytes())


if __name__ == "__main__":
    app()
