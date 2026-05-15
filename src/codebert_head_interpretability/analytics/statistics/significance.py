from collections import defaultdict

from scipy.stats import (
    f_oneway,
)

from codebert_head_interpretability.schemas.aggregated import (
    HeadMetrics,
)


class StatisticalAnalyzer:
    def entropy_anova(
        self,
        metrics: list[HeadMetrics],
        clustered,
    ):
        metric_map = {(m.layer, m.head): m for m in metrics}

        grouped = defaultdict(list)

        for c in clustered:
            grouped[c.cluster].append(metric_map[(c.layer, c.head)].entropy)

        samples = list(grouped.values())

        stat, p = f_oneway(*samples)

        return {
            "f_statistic": float(stat),
            "p_value": float(p),
        }

    def specialization_anova(
        self,
        metrics: list[HeadMetrics],
        clustered,
    ):
        metric_map = {(m.layer, m.head): m for m in metrics}

        grouped = defaultdict(list)

        for c in clustered:
            grouped[c.cluster].append(metric_map[(c.layer, c.head)].dominance_margin)

        samples = list(grouped.values())

        stat, p = f_oneway(*samples)

        return {
            "f_statistic": float(stat),
            "p_value": float(p),
        }
