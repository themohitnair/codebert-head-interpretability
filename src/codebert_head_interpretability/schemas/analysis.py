from pydantic import BaseModel


class CategoryDistribution(BaseModel):
    scores: dict[str, float]

    def to_dict(self):
        return self.scores


class HeadAnalysisResult(BaseModel):
    layer: int
    head: int
    distribution: CategoryDistribution
    entropy: float
