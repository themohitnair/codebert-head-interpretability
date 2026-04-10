from abc import ABC, abstractmethod
from codebert_head_interpretability.schemas.model_output import (
    ModelOutput,
    ModelOutputWithQuery,
)


class BaseModel(ABC):
    @abstractmethod
    def run_code(self, code: str) -> ModelOutput:
        pass

    @abstractmethod
    def run_query_code(self, query: str, code: str) -> ModelOutputWithQuery:
        pass
