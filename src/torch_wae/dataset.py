from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Generic, Iterator, Protocol, Sized, TypeVar

from pydantic.dataclasses import dataclass
from torch.utils import data


class JsonLinesDataset(data.IterableDataset):
    def __init__(self, path: Path) -> None:
        super().__init__()

        self.__path = path

    def __iter__(self) -> Iterator[Any]:
        with self.__path.open() as f:
            for line in f:
                yield json.loads(line)


@dataclass(frozen=True)
class PairJson:
    anchor: str
    positive: str
    class_id: int
    mask: bool


@dataclass(frozen=True)
class ClassificationJson:
    path: str
    class_id: int


@dataclass(frozen=True)
class ClassificationDatasetJson:
    classes: tuple[str, ...]
    examples: tuple[ClassificationJson, ...]


class ClassificationDataset(data.Dataset):
    def __init__(self, annotation: Path, root: Path) -> None:
        super().__init__()

        self.__root = root

        with annotation.open() as f:
            self.__annotations = ClassificationDatasetJson(**json.load(f))

    def __len__(self) -> int:
        return len(self.__annotations.examples)

    def __getitem__(self, index: int) -> ClassificationJson:
        return self.__annotations.examples[index]

    @property
    def root(self) -> Path:
        return self.__root

    @property
    def classes(self) -> tuple[str, ...]:
        return self.__annotations.classes


class PairDataset(data.Dataset):
    def __init__(self, annotation: Path, root: Path) -> None:
        super().__init__()

        self.__root = root

        with annotation.open() as f:
            self.__seq_line = tuple(line.strip() for line in f)

    def __len__(self) -> int:
        return len(self.__seq_line)

    def __getitem__(self, index: int) -> PairJson:
        return PairJson(**json.loads(self.__seq_line[index]))

    @property
    def root(self) -> Path:
        return self.__root


T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co], Sized):
    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> T_co: ...


class IterableDataset(Protocol[T_co]):
    def __iter__(self) -> Iterator[T_co]: ...


T = TypeVar("T")
S = TypeVar("S")
D = TypeVar("D", bound=Dataset)
D_i = TypeVar("D_i", bound=IterableDataset)


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


class TransformedIterableDataset(data.IterableDataset, Generic[D_i, T, S]):
    def __init__(
        self,
        dataset: D_i,
        transform: Callable[[T], S],
    ) -> None:
        self.__dataset = dataset
        self.__transform = transform

    def __iter__(self) -> Iterator[S]:
        for x in self.__dataset:
            yield self.__transform(x)


def transformed(
    dataset: D,
    transform: Callable[[D, int], S],
) -> TransformedDataset[D, S]:
    return TransformedDataset(dataset, transform)


def transformed_iter(
    dataset: D_i,
    transform: Callable[[T], S],
) -> TransformedIterableDataset[D_i, T, S]:
    return TransformedIterableDataset(dataset, transform)
