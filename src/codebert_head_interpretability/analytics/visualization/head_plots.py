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

    def plot_dominant_category_heatmap(
        self,
        metrics: list[HeadMetrics],
        save_path=None,
    ):
        categories = sorted(
            {
                max(
                    m.scores,
                    key=m.scores.get,
                )
                for m in metrics
            }
        )

        cat_to_idx = {cat: i for i, cat in enumerate(categories)}

        grid = np.zeros((self.layers, self.heads))

        for m in metrics:
            dominant = max(
                m.scores,
                key=m.scores.get,
            )

            grid[m.layer, m.head] = cat_to_idx[dominant]

        plt.figure(figsize=(12, 6))

        im = plt.imshow(
            grid,
            aspect="auto",
        )

        cbar = plt.colorbar(im)

        cbar.set_ticks(range(len(categories)))

        cbar.set_ticklabels(categories)

        plt.xlabel("Head")

        plt.ylabel("Layer")

        plt.title("Dominant Category per Attention Head")

        plt.xticks(range(self.heads))

        plt.yticks(range(self.layers))

        for layer in range(self.layers):
            for head in range(self.heads):
                cat_idx = int(grid[layer, head])

                cat = categories[cat_idx]

                short = cat[:3]

                plt.text(
                    head,
                    layer,
                    short,
                    ha="center",
                    va="center",
                    fontsize=6,
                )

        plt.tight_layout()

        self._show_or_save_plot(save_path)
