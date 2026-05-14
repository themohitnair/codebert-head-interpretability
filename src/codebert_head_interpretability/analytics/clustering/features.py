from codebert_head_interpretability.schemas.aggregated import (
    HeadMetrics,
)

from codebert_head_interpretability.schemas.clustering import (
    HeadFeatureVector,
)


class HeadFeatureExtractor:
    def extract(
        self,
        metrics: list[HeadMetrics],
    ) -> list[HeadFeatureVector]:
        categories = self._collect_categories(metrics)

        vectors = []

        for m in metrics:
            vector = [m.scores.get(cat, 0.0) for cat in categories]

            vectors.append(
                HeadFeatureVector(
                    layer=m.layer,
                    head=m.head,
                    vector=vector,
                    categories=categories,
                )
            )

        return vectors

    def _collect_categories(
        self,
        metrics: list[HeadMetrics],
    ) -> list[str]:
        categories = set()

        for m in metrics:
            categories.update(m.scores.keys())

        return sorted(categories)
