from pydantic import BaseModel


class HeadFeatureVector(BaseModel):
    layer: int
    head: int
    vector: list[float]
    categories: list[str]


class ClusteredHead(BaseModel):
    layer: int
    head: int
    cluster: int
    embedding_x: float
    embedding_y: float
    dominant_category: str
    specialization_score: float
