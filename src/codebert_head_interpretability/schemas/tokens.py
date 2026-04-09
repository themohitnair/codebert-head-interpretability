from pydantic import BaseModel


class ASTToken(BaseModel):
    token: str
    start: int
    end: int
    node_type: str


class ClassifiedToken(ASTToken):
    category: str


class ModelToken(BaseModel):
    text: str
    start: int
    end: int
    index: int


class AlignedToken(BaseModel):
    text: str
    start: int
    end: int
    index: int
    category: str
