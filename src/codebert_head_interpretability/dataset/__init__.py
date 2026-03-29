from .base import BaseDataset
from .codesearchnet import CodeSearchNetDataset


def get_dataset(name: str, **kwargs) -> BaseDataset:

    if name == "codesearchnet":
        return CodeSearchNetDataset(**kwargs)

    raise ValueError(f"Unknown dataset: {name}")
