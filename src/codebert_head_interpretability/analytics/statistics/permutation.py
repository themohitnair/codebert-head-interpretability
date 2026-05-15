import numpy as np

from codebert_head_interpretability.schemas.aggregated import (
    LayerMetrics,
)


class PermutationTester:
    def entropy_trend_test(
        self,
        metrics: list[LayerMetrics],
        n_perm=5000,
    ):
        layers = np.array([m.layer for m in metrics])

        entropy = np.array([m.entropy for m in metrics])

        observed = np.corrcoef(
            layers,
            entropy,
        )[0, 1]

        count = 0

        for _ in range(n_perm):
            shuffled = np.random.permutation(entropy)

            corr = np.corrcoef(
                layers,
                shuffled,
            )[0, 1]

            if abs(corr) >= abs(observed):
                count += 1

        p_value = count / n_perm

        return {
            "observed_correlation": float(observed),
            "p_value": float(p_value),
        }
