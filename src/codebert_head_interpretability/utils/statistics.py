import numpy as np
from scipy.spatial.distance import jensenshannon


EPS = 1e-12


def normalize_distribution(
    dist: dict[str, float],
):
    total = sum(dist.values())

    if total <= 0:
        return dist

    return {k: v / total for k, v in dist.items()}


def distribution_vector(
    dist: dict[str, float],
    categories: list[str],
):
    vec = np.array(
        [dist.get(cat, 0.0) for cat in categories],
        dtype=np.float64,
    )

    vec += EPS

    vec /= vec.sum()

    return vec


def js_divergence(
    p: dict[str, float],
    q: dict[str, float],
):
    categories = sorted(set(p.keys()) | set(q.keys()))

    p_vec = distribution_vector(
        p,
        categories,
    )

    q_vec = distribution_vector(
        q,
        categories,
    )

    return float(
        jensenshannon(
            p_vec,
            q_vec,
            base=2,
        )
    )
