from typing import Iterator
from datasets import Dataset
from .base import BaseDataset
from codebert_head_interpretability.models.code_query import CodeQueryModel


class CodeSearchNetDataset(BaseDataset):
    def __init__(self, cache_dir="./hf_cache", language="python"):
        super().__init__(
            dataset_name="code-search-net/code_search_net", cache_dir=cache_dir
        )
        self.language = language

    def to_examples(self, dataset: Dataset) -> Iterator[CodeQueryModel]:
        for row in dataset:
            if row["language"] != self.language:
                continue

            code: str = row["func_code_string"]
            query: str = row["func_documentation_string"]

            code = code.strip()
            query = query.strip()

            if not code or not query:
                continue

            yield CodeQueryModel(code=code, query=query)
