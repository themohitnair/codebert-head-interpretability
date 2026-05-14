from collections import defaultdict

from codebert_head_interpretability.schemas.aggregated import (
    HeadMetrics,
    LayerMetrics,
)


class LayerMetricsAggregator:
    def aggregate(
        self,
        heads: list[HeadMetrics],
    ) -> list[LayerMetrics]:
        grouped: defaultdict[int, list[HeadMetrics]] = defaultdict(list)

        for head in heads:
            grouped[head.layer].append(head)

        results: list[LayerMetrics] = []

        for layer, layer_heads in sorted(grouped.items()):
            semanticity = self._average_category(
                layer_heads,
                "identifier",  # Identifiers are semantic
            )

            structurality = self._average_category(
                layer_heads,
                "bracket",  # Brackets are structural
            )

            entropy = sum(h.entropy for h in layer_heads) / len(layer_heads)

            dominant_category = self._dominant_category(layer_heads)

            results.append(
                LayerMetrics(
                    layer=layer,
                    semanticity=semanticity,
                    structurality=structurality,
                    entropy=entropy,
                    dominant_category=dominant_category,
                )
            )

        return results

    def _average_category(
        self,
        heads: list[HeadMetrics],
        category: str,
    ) -> float:
        total = 0.0

        for head in heads:
            total += head.scores.get(category, 0.0)

        return total / len(heads)

    def _dominant_category(
        self,
        heads: list[HeadMetrics],
    ) -> str:
        totals: defaultdict[str, float] = defaultdict(float)

        for head in heads:
            for cat, val in head.scores.items():
                totals[cat] += val

        return max(totals, key=totals.get)
