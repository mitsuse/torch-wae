from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer()


@app.command()
def main(
    annotation: Path = typer.Option(
        ...,
        help="the path of a dataset to be encoded.",
    ),
    output: Path = typer.Option(
        ...,
        help="the output path of a converted JSONL file.,",
    ),
) -> None:
    import json

    from tqdm import tqdm

    from torch_wae.dataset import ClassificationDataset

    dataset = ClassificationDataset(
        annotation=annotation,
        root=annotation.parent,
    )

    n = len(dataset)

    output.parent.mkdir(parents=True, exist_ok=True)
    with tqdm(total=n) as progress, output.open(mode="w") as f:
        for i in range(n):
            example = dataset[i]
            line = json.dumps(
                {
                    "path": example.path,
                    "class_id": example.class_id,
                }
            )
            f.write(f"{line}\n")
            progress.update(1)


if __name__ == "__main__":
    app()
