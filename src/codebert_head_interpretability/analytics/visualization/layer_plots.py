import os

import matplotlib.pyplot as plt

from codebert_head_interpretability.schemas.aggregated import (
    LayerMetrics,
)


class LayerPlots:
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

    def plot_semantic_vs_structural(
        self,
        metrics: list[LayerMetrics],
        save_path=None,
    ):
        layers = [m.layer for m in metrics]

        semanticity = [m.semanticity for m in metrics]

        structurality = [m.structurality for m in metrics]

        plt.figure(figsize=(10, 5))

        plt.plot(
            layers,
            semanticity,
            marker="o",
            label="Semantic",
        )

        plt.plot(
            layers,
            structurality,
            marker="o",
            label="Syntactic",
        )

        plt.xlabel("Layer")

        plt.ylabel("Average Attention")

        plt.title("Semantic vs Syntactic Attention")

        plt.legend()

        plt.grid(True)

        plt.tight_layout()

        self._show_or_save_plot(save_path)

    def plot_layer_entropy(
        self,
        metrics: list[LayerMetrics],
        save_path=None,
    ):
        layers = [m.layer for m in metrics]

        entropies = [m.entropy for m in metrics]

        plt.figure(figsize=(10, 5))

        plt.plot(
            layers,
            entropies,
            marker="o",
        )

        plt.xlabel("Layer")

        plt.ylabel("Entropy")

        plt.title("Average Layer Entropy")

        plt.grid(True)

        plt.tight_layout()

        self._show_or_save_plot(save_path)
