from typing import Any
from pydantic import BaseModel
from .tokens import ModelToken


class WindowOutput(BaseModel):
    tokens: list[ModelToken]
    attentions: Any


class ModelOutput(BaseModel):
    windows: list[WindowOutput]
