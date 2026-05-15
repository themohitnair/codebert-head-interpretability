import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from codebert_head_interpretability.schemas.aggregated import HeadMetrics
from codebert_head_interpretability.schemas.clustering import ClusteredHead


class ClusterPlots:
    def __init__(self, layers: int = 12, heads: int = 12):
        self.layers = layers
        self.heads = heads

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

    def plot_cluster_heatmap(
        self,
        clustered: list[ClusteredHead],
        save_path=None,
    ):
        grid = np.zeros((self.layers, self.heads))

        for c in clustered:
            grid[c.layer, c.head] = c.cluster

        plt.figure(figsize=(12, 6))

        im = plt.imshow(
            grid,
            aspect="auto",
        )

        cbar = plt.colorbar(im)

        cbar.set_label("Cluster ID")

        plt.xlabel("Head")

        plt.ylabel("Layer")

        plt.title("Cluster Assignment per Attention Head")

        plt.xticks(range(self.heads))

        plt.yticks(range(self.layers))

        for c in clustered:
            plt.text(
                c.head,
                c.layer,
                str(c.cluster),
                ha="center",
                va="center",
                fontsize=7,
            )

        plt.tight_layout()

        self._show_or_save_plot(save_path)

    def plot_cluster_distribution_by_layer(
        self,
        clustered: list[ClusteredHead],
        save_path=None,
    ):
        cluster_ids = sorted({c.cluster for c in clustered})

        counts = {cluster: np.zeros(self.layers) for cluster in cluster_ids}

        for c in clustered:
            counts[c.cluster][c.layer] += 1

        plt.figure(figsize=(10, 5))

        for cluster in cluster_ids:
            plt.plot(
                range(self.layers),
                counts[cluster],
                marker="o",
                label=f"Cluster {cluster}",
            )

        plt.xlabel("Layer")

        plt.ylabel("Number of Heads")

        plt.title("Cluster Distribution Across Layers")

        plt.legend()

        plt.grid(True)

        plt.tight_layout()

        self._show_or_save_plot(save_path)

    def plot_cluster_entropy(
        self,
        clustered: list[ClusteredHead],
        metrics: list[HeadMetrics],
        save_path=None,
    ):
        metric_map = {(m.layer, m.head): m for m in metrics}

        grouped = defaultdict(list)

        for c in clustered:
            grouped[c.cluster].append(metric_map[(c.layer, c.head)])

        cluster_ids = []
        entropies = []

        for cluster, heads in sorted(grouped.items()):
            avg_entropy = sum(h.entropy for h in heads) / len(heads)

            cluster_ids.append(cluster)

            entropies.append(avg_entropy)

        plt.figure(figsize=(8, 5))

        plt.bar(
            range(len(cluster_ids)),
            entropies,
        )

        plt.xticks(
            range(len(cluster_ids)),
            [f"C{c}" for c in cluster_ids],
        )

        plt.ylabel("Average Entropy")

        plt.title("Entropy per Cluster")

        plt.tight_layout()

        self._show_or_save_plot(save_path)

    def plot_cluster_specialization(
        self,
        clustered: list[ClusteredHead],
        metrics: list[HeadMetrics],
        save_path=None,
    ):
        metric_map = {(m.layer, m.head): m for m in metrics}

        grouped = defaultdict(list)

        for c in clustered:
            grouped[c.cluster].append(metric_map[(c.layer, c.head)])

        cluster_ids = []
        specialization = []

        for cluster, heads in sorted(grouped.items()):
            avg_spec = sum(h.dominance_margin for h in heads) / len(heads)

            cluster_ids.append(cluster)

            specialization.append(avg_spec)

        plt.figure(figsize=(8, 5))

        plt.bar(
            range(len(cluster_ids)),
            specialization,
        )

        plt.xticks(
            range(len(cluster_ids)),
            [f"C{c}" for c in cluster_ids],
        )

        plt.ylabel("Average Specialization")

        plt.title("Specialization Strength per Cluster")

        plt.tight_layout()

        self._show_or_save_plot(save_path)
