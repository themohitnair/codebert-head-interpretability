import os

import matplotlib.pyplot as plt
import numpy as np

from codebert_head_interpretability.schemas.aggregated import (
    HeadMetrics,
)


class HeadPlots:
    def __init__(self, layers=12, heads=12):
        self.layers = layers
        self.heads = heads

    def _show_or_save_plot(self, save_path):
        if save_path:
            os.makedirs(
                os.path.dirname(save_path),
                exist_ok=True,
            )

            plt.savefig(
                save_path,
                dpi=300,
                bbox_inches="tight",
            )

            plt.close()

        else:
            plt.show()

    def _build_grid(
        self,
        metrics: list[HeadMetrics],
        value_fn,
    ):
        grid = np.zeros((self.layers, self.heads))

        for m in metrics:
            grid[m.layer, m.head] = value_fn(m)

        return grid

    def plot_category_heatmap(
        self,
        metrics: list[HeadMetrics],
        category: str,
        save_path=None,
    ):
        heatmap = self._build_grid(
            metrics,
            lambda m: m.scores.get(category, 0),
        )

        plt.figure(figsize=(10, 6))

        im = plt.imshow(
            heatmap,
            aspect="auto",
        )

        plt.colorbar(
            im,
            label=f"{category} attention",
        )

        plt.xlabel("Head")
        plt.ylabel("Layer")

        plt.title(f"Attention Heatmap: {category}")

        plt.xticks(range(self.heads))
        plt.yticks(range(self.layers))

        plt.tight_layout()

        self._show_or_save_plot(save_path)

    def plot_entropy(
        self,
        metrics: list[HeadMetrics],
        save_path=None,
    ):
        labels = []
        values = []

        for m in sorted(
            metrics,
            key=lambda x: (x.layer, x.head),
        ):
            labels.append(f"L{m.layer}H{m.head}")

            values.append(m.entropy)

        plt.figure(figsize=(16, 5))

        plt.bar(
            range(len(values)),
            values,
        )

        plt.xticks(
            range(len(labels)),
            labels,
            rotation=90,
        )

        plt.ylabel("Entropy")

        plt.title("Entropy per Head")

        plt.tight_layout()

        self._show_or_save_plot(save_path)
