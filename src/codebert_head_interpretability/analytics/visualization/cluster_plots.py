import os

import matplotlib.pyplot as plt

from codebert_head_interpretability.schemas.clustering import (
    ClusteredHead,
)


class ClusterPlots:
    def _show_or_save_plot(
        self,
        save_path,
    ):
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

    def plot_pca_clusters(
        self,
        clustered: list[ClusteredHead],
        save_path=None,
    ):
        plt.figure(figsize=(10, 8))

        clusters = sorted(set(c.cluster for c in clustered))

        for cluster in clusters:
            subset = [c for c in clustered if c.cluster == cluster]

            xs = [c.embedding_x for c in subset]
            ys = [c.embedding_y for c in subset]

            plt.scatter(
                xs,
                ys,
                label=f"Cluster {cluster}",
            )

            for c in subset:
                plt.text(
                    c.embedding_x,
                    c.embedding_y,
                    f"L{c.layer}H{c.head}",
                    fontsize=8,
                )

        plt.xlabel("PCA Component 1")

        plt.ylabel("PCA Component 2")

        plt.title("Attention Head Clusters")

        plt.legend()

        plt.grid(True)

        plt.tight_layout()

        self._show_or_save_plot(save_path)
