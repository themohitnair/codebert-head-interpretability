from typing import Any
from pydantic import BaseModel
from .tokens import ModelToken


class WindowOutput(BaseModel):
    tokens: list[ModelToken]
    attentions: Any


class WindowOutputWithQuery(WindowOutput):
    query_len: int


class ModelOutput(BaseModel):
    windows: list[WindowOutput]


class ModelOutputWithQuery(BaseModel):
    windows: list[WindowOutputWithQuery]
