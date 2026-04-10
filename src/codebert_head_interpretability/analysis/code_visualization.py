import os
import numpy as np
import matplotlib.pyplot as plt

from codebert_head_interpretability.utils.maths import compute_entropy


class AttentionVisualizer:
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

    def _compute_avg_stats(self, global_stats, count_per_head):
        avg_stats = {}

        for (layer, head), cat_dict in global_stats.items():
            total = count_per_head[(layer, head)]
            avg_stats[(layer, head)] = {k: v / total for k, v in cat_dict.items()}

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

    def plot_category_heatmap(
        self,
        global_stats,
        count_per_head,
        category="identifier",
        save_path=None,
    ):
        avg_stats = self._compute_avg_stats(global_stats, count_per_head)

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

    def plot_top_category_map(
        self,
        global_stats,
        count_per_head,
        save_path=None,
    ):
        avg_stats = self._compute_avg_stats(global_stats, count_per_head)
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

    def plot_head_distribution(
        self,
        global_stats,
        count_per_head,
        save_path=None,
    ):
        avg_stats = self._compute_avg_stats(global_stats, count_per_head)
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

    def plot_entropy(
        self,
        global_stats,
        count_per_head,
        save_path=None,
    ):
        avg_stats = self._compute_avg_stats(global_stats, count_per_head)

        labels = []
        entropies = []

        for (layer, head), cat_dict in sorted(avg_stats.items()):
            labels.append(f"L{layer}H{head}")
            entropies.append(compute_entropy(cat_dict))

        plt.figure(figsize=(14, 5))
        plt.bar(range(len(entropies)), entropies)

        plt.xticks(range(len(labels)), labels, rotation=90)
        plt.ylabel("Entropy")
        plt.title("Entropy per Head")
        plt.tight_layout()

        self._show_or_save_plot(save_path)
