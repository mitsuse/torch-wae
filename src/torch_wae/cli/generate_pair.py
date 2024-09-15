from __future__ import annotations

from pathlib import Path
from random import Random

import typer

from torch_wae.dataset import ClassificationDatasetJson, ClassificationJson, PairJson

app = typer.Typer()


@app.command()
def generate_pair(
    max_pair: int = typer.Option(
        256,
        help="the maximum number of pairs per class",
    ),
    seed: int = typer.Option(
        20240820,
        help="the random seed",
    ),
    annotation: Path = typer.Option(
        ...,
        help="the path of an annotation file for classification",
    ),
) -> None:
    import dataclasses
    import json

    from tqdm import tqdm

    from torch_wae import fs

    random = Random(seed)

    with annotation.open() as f:
        dataset = ClassificationDatasetJson(**json.load(f))

    n_class = len(dataset.classes)
    n_example = len(dataset.examples)

    group_example: tuple[list[ClassificationJson], ...] = tuple(
        [] for _ in range(n_class)
    )

    with tqdm(total=n_example) as progress:
        for example in dataset.examples:
            group_example[example.class_id].append(example)

            progress.update(1)

    name = fs.basename(annotation)
    path_output = annotation.parent / f"pair-{name}.jsonl"

    with tqdm(total=n_class) as progress, path_output.open(mode="w") as f:
        for c, seq_example in enumerate(group_example):
            for a, b in generate_random_pair(random, seq_example, max_pair):
                json_pair = dataclasses.asdict(convert_to_pair((a, b)))
                f.write(json.dumps(json_pair))
                f.write("\n")

            progress.update(1)


def generate_random_pair(
    random: Random,
    seq_example: list[ClassificationJson],
    n: int,
) -> tuple[tuple[ClassificationJson, ClassificationJson], ...]:
    n_example = len(seq_example)
    if n_example < 2:
        return tuple()

    set_pair: set[tuple[ClassificationJson, ClassificationJson]] = set()
    for _ in range(n):
        a, b = tuple(random.sample(seq_example, 2))
        if (a, b) not in set_pair and (b, a) not in set_pair:
            set_pair.add((a, b))

    return tuple(set_pair)


def convert_to_pair(
    pair: tuple[ClassificationJson, ClassificationJson],
) -> PairJson:
    anchor, positive = pair
    assert anchor.class_id == positive.class_id

    return PairJson(
        anchor=anchor.path,
        positive=positive.path,
        class_id=anchor.class_id,
        mask=False,
    )
