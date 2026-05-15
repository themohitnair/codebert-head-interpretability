from abc import ABC, abstractmethod
from typing import Iterator
from datasets import Dataset, load_dataset

from codebert_head_interpretability.schemas.code_query import CodeQueryModel


class BaseDataset(ABC):
    def __init__(self, dataset_name: str, cache_dir: str = "./hf_cache"):
        self.dataset_name = dataset_name
        self.cache_dir = cache_dir

    def load(self, split: str, subset: str | None = None) -> Dataset:
        ds = load_dataset(self.dataset_name, subset, cache_dir=self.cache_dir)
        return ds[split]

    @abstractmethod
    def to_examples(
        self, dataset: Dataset, max_examples: int | None = None
    ) -> Iterator[CodeQueryModel]:
        pass

    @property
    @abstractmethod
    def language(self) -> str:
        pass
