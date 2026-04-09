from pydantic import BaseModel
from typing import Any, List
from .tokens import ModelToken


class ModelOutput(BaseModel):
    tokens: List[ModelToken]
    attentions: Any
