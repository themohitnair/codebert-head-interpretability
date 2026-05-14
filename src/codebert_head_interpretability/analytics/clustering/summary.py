from collections import defaultdict

from codebert_head_interpretability.schemas.clustering import (
    ClusteredHead,
)

from codebert_head_interpretability.schemas.aggregated import (
    HeadMetrics,
)


class ClusterSummary:
    def summarize(
        self,
        clustered: list[ClusteredHead],
        metrics: list[HeadMetrics],
    ):
        metric_map = {(m.layer, m.head): m for m in metrics}

        grouped: dict[int, list[ClusteredHead]] = defaultdict(list)

        for c in clustered:
            grouped[c.cluster].append(c)

        summaries: dict[int, dict] = {}

        for cluster, heads in grouped.items():
            totals = defaultdict(float)

            for h in heads:
                metric = metric_map[(h.layer, h.head)]

                for cat, val in metric.scores.items():
                    totals[cat] += val

            n = len(heads)

            avg_scores = {cat: val / n for cat, val in totals.items()}

            summaries[cluster] = {
                "size": n,
                "dominant_category": max(
                    avg_scores,
                    key=avg_scores.get,
                ),
                "scores": avg_scores,
            }

        return summaries
