import os
import numpy as np
import matplotlib.pyplot as plt

from codebert_head_interpretability.schemas.analysis import HeadAnalysisResult
from codebert_head_interpretability.utils.maths import compute_entropy


class HeadAnalysisVisualizer:
    def __init__(self, layers=12, heads=12):
        self.layers = layers
        self.heads = heads

    def _show_or_save_plot(self, save_path):
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def _group_by_head(self, results: list[HeadAnalysisResult]):
        grouped = {}

        for r in results:
            key = (r.layer, r.head)

            if key not in grouped:
                grouped[key] = []

            grouped[key].append(r.distribution.scores)

        return grouped

    def _average_distributions(self, grouped):
        avg_stats = {}

        for key, dist_list in grouped.items():
            agg = {}

            for dist in dist_list:
                for cat, val in dist.items():
                    agg[cat] = agg.get(cat, 0) + val

            total = len(dist_list)
            avg_stats[key] = {k: v / total for k, v in agg.items()}

        return avg_stats

    def _get_categories(self, avg_stats):
        categories = set()
        for cat_dict in avg_stats.values():
            categories.update(cat_dict.keys())
        return sorted(categories)

    def _build_grid(self, avg_stats, value_fn):
        grid = np.zeros((self.layers, self.heads))

        for (layer, head), cat_dict in avg_stats.items():
            grid[layer, head] = value_fn(cat_dict)

        return grid

    # ===================== HEATMAP =====================

    def plot_category_heatmap(self, results, category="identifier", save_path=None):
        grouped = self._group_by_head(results)
        avg_stats = self._average_distributions(grouped)

        def value_fn(cat_dict):
            return cat_dict.get(category, 0)

        heatmap = self._build_grid(avg_stats, value_fn)

        plt.figure(figsize=(10, 6))
        im = plt.imshow(heatmap, aspect="auto")
        plt.colorbar(im, label=f"{category} attention")

        plt.xlabel("Head")
        plt.ylabel("Layer")
        plt.title(f"Attention Heatmap: {category}")
        plt.xticks(range(self.heads))
        plt.yticks(range(self.layers))
        plt.tight_layout()

        self._show_or_save_plot(save_path)

    # ===================== TOP CATEGORY =====================

    def plot_top_category_map(self, results, save_path=None):
        grouped = self._group_by_head(results)
        avg_stats = self._average_distributions(grouped)

        categories = self._get_categories(avg_stats)
        cat_to_idx = {cat: i for i, cat in enumerate(categories)}

        def value_fn(cat_dict):
            top_cat = max(cat_dict, key=cat_dict.get)
            return cat_to_idx[top_cat]

        grid = self._build_grid(avg_stats, value_fn)

        plt.figure(figsize=(10, 6))
        im = plt.imshow(grid, aspect="auto")

        cbar = plt.colorbar(im)
        cbar.set_ticks(range(len(categories)))
        cbar.set_ticklabels(categories)

        plt.xlabel("Head")
        plt.ylabel("Layer")
        plt.title("Top Category per Head")
        plt.xticks(range(self.heads))
        plt.yticks(range(self.layers))
        plt.tight_layout()

        self._show_or_save_plot(save_path)

    # ===================== DISTRIBUTION =====================

    def plot_head_distribution(self, results, save_path=None):
        grouped = self._group_by_head(results)
        avg_stats = self._average_distributions(grouped)

        categories = self._get_categories(avg_stats)

        labels = []
        data = {cat: [] for cat in categories}

        for (layer, head), cat_dict in sorted(avg_stats.items()):
            labels.append(f"L{layer}H{head}")

            for cat in categories:
                data[cat].append(cat_dict.get(cat, 0))

        bottom = np.zeros(len(labels))

        plt.figure(figsize=(16, 6))

        for cat in categories:
            values = np.array(data[cat])
            plt.bar(range(len(labels)), values, bottom=bottom, label=cat)
            bottom += values

        plt.xticks(range(len(labels)), labels, rotation=90)
        plt.ylabel("Attention Distribution")
        plt.title("Category Distribution per Head")
        plt.legend()
        plt.tight_layout()

        self._show_or_save_plot(save_path)

    # ===================== ENTROPY =====================

    def plot_entropy(self, results: list[HeadAnalysisResult], save_path=None):
        grouped = self._group_by_head(results)

        labels = []
        entropies = []

        for (layer, head), dist_list in sorted(grouped.items()):
            avg_dist = {}

            for dist in dist_list:
                for cat, val in dist.items():
                    avg_dist[cat] = avg_dist.get(cat, 0) + val

            total = len(dist_list)
            avg_dist = {k: v / total for k, v in avg_dist.items()}

            entropy = compute_entropy(avg_dist)

            labels.append(f"L{layer}H{head}")
            entropies.append(entropy)

        plt.figure(figsize=(14, 5))
        plt.bar(range(len(entropies)), entropies)

        plt.xticks(range(len(labels)), labels, rotation=90)
        plt.ylabel("Entropy")
        plt.title("Entropy per Head")
        plt.tight_layout()

        self._show_or_save_plot(save_path)
