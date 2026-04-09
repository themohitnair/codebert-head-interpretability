from pydantic import BaseModel


class CodeQueryModel(BaseModel):
    code: str
    query: str
