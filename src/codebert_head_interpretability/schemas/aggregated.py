from pydantic import BaseModel


class LayerMetrics(BaseModel):
    layer: int
    semanticity: float
    structurality: float
    entropy: float
    dominant_category: str


class HeadMetrics(BaseModel):
    layer: int
    head: int
    scores: dict[str, float]
    entropy: float
    dominance_margin: float
