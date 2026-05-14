import numpy as np

from sklearn.cluster import KMeans

from codebert_head_interpretability.schemas.aggregated import (
    HeadMetrics,
)

from codebert_head_interpretability.schemas.clustering import (
    ClusteredHead,
    HeadFeatureVector,
)


class HeadClusterAnalyzerKMeans:
    def cluster(
        self,
        vectors: list[HeadFeatureVector],
        metrics: list[HeadMetrics],
        embeddings,
        n_clusters=4,
    ) -> list[ClusteredHead]:
        X = np.array([v.vector for v in vectors])

        model = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init="auto",
        )

        labels = model.fit_predict(X)

        clustered = []

        for i, label in enumerate(labels):
            metric = metrics[i]

            clustered.append(
                ClusteredHead(
                    layer=metric.layer,
                    head=metric.head,
                    cluster=int(label),
                    embedding_x=float(embeddings[i][0]),
                    embedding_y=float(embeddings[i][1]),
                    dominant_category=max(
                        metric.scores,
                        key=metric.scores.get,
                    ),
                    specialization_score=(metric.dominance_margin),
                )
            )

        return clustered
