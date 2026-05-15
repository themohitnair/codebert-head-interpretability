from collections import defaultdict

from codebert_head_interpretability.schemas.aggregated import (
    HeadMetrics,
)

from codebert_head_interpretability.utils.statistics import (
    js_divergence,
)


class ClusterDivergenceAnalyzer:
    def cluster_average_distributions(
        self,
        metrics: list[HeadMetrics],
        clustered,
    ):
        metric_map = {(m.layer, m.head): m for m in metrics}

        grouped = defaultdict(list)

        for c in clustered:
            grouped[c.cluster].append(metric_map[(c.layer, c.head)])

        averages = {}

        for cluster, heads in grouped.items():
            totals = defaultdict(float)

            for h in heads:
                for cat, val in h.scores.items():
                    totals[cat] += val

            n = len(heads)

            averages[cluster] = {cat: val / n for cat, val in totals.items()}

        return averages

    def pairwise_js_divergence(
        self,
        metrics,
        clustered,
    ):
        averages = self.cluster_average_distributions(
            metrics,
            clustered,
        )

        cluster_ids = sorted(averages.keys())

        results = {}

        for i in range(len(cluster_ids)):
            for j in range(
                i + 1,
                len(cluster_ids),
            ):
                c1 = cluster_ids[i]
                c2 = cluster_ids[j]

                div = js_divergence(
                    averages[c1],
                    averages[c2],
                )

                results[(c1, c2)] = div

        return results
