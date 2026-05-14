from collections import defaultdict

from codebert_head_interpretability.schemas.analysis import HeadAnalysisResult
from codebert_head_interpretability.schemas.aggregated import HeadMetrics


class HeadMetricsAggregator:
    def aggregate(
        self,
        results: list[HeadAnalysisResult],
    ) -> list[HeadMetrics]:
        grouped: dict[tuple[int, int], list[HeadAnalysisResult]] = defaultdict(list)

        for r in results:
            grouped[(r.layer, r.head)].append(r)

        aggregated: list[HeadMetrics] = []

        for (layer, head), items in grouped.items():
            avg_scores = self._average_scores(items)

            entropy = sum(x.entropy for x in items) / len(items)

            dominance_margin = self._dominance_margin(avg_scores)

            aggregated.append(
                HeadMetrics(
                    layer=layer,
                    head=head,
                    scores=avg_scores,
                    entropy=entropy,
                    dominance_margin=dominance_margin,
                )
            )

        return aggregated

    def _average_scores(self, items: list[HeadAnalysisResult]) -> dict[str, float]:
        totals = defaultdict(float)

        for item in items:
            for cat, val in item.distribution.scores.items():
                totals[cat] += val

        n = len(items)

        return {cat: val / n for cat, val in totals.items()}

    def _dominance_margin(self, scores: dict[str, float]) -> float:
        vals = sorted(scores.values(), reverse=True)

        if len(vals) < 2:
            return 0.0

        return vals[0] - vals[1]
