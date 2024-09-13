from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Generic, Protocol, Sized, TypeVar

from pydantic.dataclasses import dataclass
from torch.utils import data


@dataclass(frozen=True)
class PairJson:
    anchor: str
    positive: str
    class_id: int
    mask: bool


@dataclass(frozen=True)
class PairDatasetJson:
    examples: tuple[PairJson, ...]


@dataclass(frozen=True)
class ClassificationJson:
    path: str
    class_id: int


@dataclass(frozen=True)
class ClassificationDatasetJson:
    classes: tuple[str, ...]
    examples: tuple[ClassificationJson, ...]


class JsonLinesDataset(data.Dataset):
    def __init__(self, annotation: Path, root: Path) -> None:
        super().__init__()

        self.__root = root

        with annotation.open() as f:
            self.__seq_line = tuple(line.strip() for line in f)

    def __len__(self) -> int:
        return len(self.__seq_line)

    def __getitem__(self, index: int) -> Any:
        import json

        return json.loads(self.__seq_line[index])

    @property
    def root(self) -> Path:
        return self.__root


def load_classification_dataset(
    annotation: Path,
    root: Path,
) -> Dataset[ClassificationJson]:
    return transformed(
        JsonLinesDataset(annotation, root),
        transformed_to_classification,
    )


def load_pair_dataset(
    annotation: Path,
    root: Path,
) -> Dataset[PairJson]:
    return transformed(
        JsonLinesDataset(annotation, root),
        transformed_to_pair,
    )


def transformed_to_classification(example: Any, index: int) -> ClassificationJson:
    return ClassificationJson(**example)


def transformed_to_pair(example: Any, index: int) -> PairJson:
    return PairJson(**example)


T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co], Sized):
    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> T_co: ...


S = TypeVar("S")
D = TypeVar("D", bound=Dataset)


class TransformedDataset(data.Dataset, Generic[D, S]):
    def __init__(
        self,
        dataset: D,
        transform: Callable[[D, int], S],
    ) -> None:
        self.__dataset = dataset
        self.__transform = transform

    def __len__(self) -> int:
        return len(self.__dataset)

    def __getitem__(self, index: int) -> S:
        dataset = self.__dataset
        transform = self.__transform
        return transform(dataset, index)


def transformed(
    dataset: D,
    transform: Callable[[D, int], S],
) -> TransformedDataset[D, S]:
    return TransformedDataset(dataset, transform)
