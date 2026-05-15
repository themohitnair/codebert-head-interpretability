import numpy as np

from sklearn.metrics import (
    silhouette_score,
)

from codebert_head_interpretability.schemas.clustering import (
    HeadFeatureVector,
    ClusteredHead,
)


class ClusterValidator:
    def silhouette(
        self,
        vectors: list[HeadFeatureVector],
        clustered: list[ClusteredHead],
    ):
        X = np.array([v.vector for v in vectors])

        labels = np.array([c.cluster for c in clustered])

        return float(
            silhouette_score(
                X,
                labels,
            )
        )
