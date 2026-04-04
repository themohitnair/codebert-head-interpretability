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

    def to_examples(
        self, dataset: Dataset, max_examples: int | None = None
    ) -> Iterator[CodeQueryModel]:
        n = 0
        for row in dataset:
            n += 1
            if max_examples is not None and n > max_examples:
                break

            if row["language"] != self.language:  # type: ignore
                continue

            code: str = row["func_code_string"]  # type: ignore
            query: str = row["func_documentation_string"]  # type: ignore

            code = self.cleanup_code(code)
            query = query.strip()

            if not code or not query:
                continue

            yield CodeQueryModel(code=code, query=query)

    def cleanup_code(self, code: str, keep_comments: bool = False) -> str:
        """Remove comments and docstrings from the code."""
        lines = code.splitlines()
        cleaned_lines = []
        in_docstring = False
        for line in lines:
            stripped_line = line.strip()
            if stripped_line.startswith('"""') or stripped_line.startswith("'''"):
                in_docstring = not in_docstring
                continue
            if in_docstring:
                continue
            if not keep_comments and stripped_line.startswith("#"):
                continue
            cleaned_lines.append(line)
        return "\n".join(cleaned_lines)
