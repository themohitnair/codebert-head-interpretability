from pydantic import BaseModel


class ASTToken(BaseModel):
    token: str
    start: int
    end: int
    node_type: str


class ClassifiedToken(ASTToken):
    category: str


class BPEToken(BaseModel):
    text: str
    start: int
    end: int
    token_id: int


class AlignedToken(BaseModel):
    text: str
    start: int
    end: int
    token_id: int
    category: str
