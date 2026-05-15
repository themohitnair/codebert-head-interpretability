import os

from codebert_head_interpretability.analytics.analysis.codebert import (
    HeadAnalysisAnalyzer,
)
from codebert_head_interpretability.datasets.base import BaseDataset
from codebert_head_interpretability.models.base import BaseModel
from codebert_head_interpretability.parsers.token_classifier import TokenClassifier
from codebert_head_interpretability.parsers.tree_sitter_parser import CodeParser
from codebert_head_interpretability.schemas.analysis import HeadAnalysisResult
from codebert_head_interpretability.schemas.code_query import CodeQueryModel
from codebert_head_interpretability.analytics.aggregation.head import (
    HeadMetricsAggregator,
)
from codebert_head_interpretability.analytics.aggregation.layer import (
    LayerMetricsAggregator,
)
from codebert_head_interpretability.analytics.visualization.head_plots import HeadPlots
from codebert_head_interpretability.analytics.visualization.layer_plots import (
    LayerPlots,
)
from codebert_head_interpretability.analytics.clustering.features import (
    HeadFeatureExtractor,
)
from codebert_head_interpretability.analytics.clustering.pca import PCAEmbedder
from codebert_head_interpretability.analytics.clustering.kmeans import (
    HeadClusterAnalyzerKMeans,
)
from codebert_head_interpretability.analytics.clustering.summary import ClusterSummary
from codebert_head_interpretability.analytics.visualization.cluster_plots import (
    ClusterPlots,
)
from codebert_head_interpretability.analytics.clustering.validation import (
    ClusterValidator,
)
from codebert_head_interpretability.analytics.clustering.divergence import (
    ClusterDivergenceAnalyzer,
)
from codebert_head_interpretability.analytics.statistics.significance import (
    StatisticalAnalyzer,
)
from codebert_head_interpretability.analytics.statistics.permutation import (
    PermutationTester,
)


class BasePipeline:
    def __init__(self, dataset: BaseDataset, model: BaseModel):
        self.dataset = dataset
        self.model = model
        self.parser = CodeParser(language=dataset.language)
        self.analyzer = HeadAnalysisAnalyzer()
        self.classifier = TokenClassifier(parser=self.parser)
        self.head_aggregator = HeadMetricsAggregator()
        self.layer_aggregator = LayerMetricsAggregator()
        self.head_plots = HeadPlots()
        self.layer_plots = LayerPlots()
        self.feature_extractor = HeadFeatureExtractor()
        self.pca_embedder = PCAEmbedder()
        self.cluster_analyzer = HeadClusterAnalyzerKMeans()
        self.cluster_summary = ClusterSummary()
        self.cluster_plots = ClusterPlots()
        self.cluster_validator = ClusterValidator()
        self.cluster_divergence = ClusterDivergenceAnalyzer()
        self.stats_analyzer = StatisticalAnalyzer()
        self.permutation_tester = PermutationTester()

    def process_example(self, example: CodeQueryModel) -> list[HeadAnalysisResult]:
        raise NotImplementedError

    def run(self, split="train", max_examples=100, output_dir="outputs"):
        ds = self.dataset.load(split, self.dataset.language)

        all_results: list[HeadAnalysisResult] = []

        print("\nProcessing dataset...\n")

        for i, example in enumerate(
            self.dataset.to_examples(ds, max_examples=max_examples)
        ):
            try:
                results = self.process_example(example)
                all_results.extend(results)

            except Exception as e:
                print(f"Skipping example {i}: {e}")
                continue

            if i % 10 == 0 and i > 0:
                print(f"Processed {i} examples...")

        print("\nGenerating visualizations...\n")

        self._visualize(all_results, output_dir)

        print(f"\nAll outputs saved to '{output_dir}/'\n")

    def _save_stats_to_file(self, output_dir: str, stats_content: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        stats_path = os.path.join(output_dir, "analysis_stats.txt")
        with open(stats_path, "w") as f:
            f.write(stats_content)
        print(f"Stats saved to '{stats_path}'")

    def _visualize(
        self,
        results: list[HeadAnalysisResult],
        output_dir: str,
    ):
        if not results:
            print("No results to visualize.")
            return

        head_metrics = self.head_aggregator.aggregate(results)

        layer_metrics = self.layer_aggregator.aggregate(head_metrics)

        categories = set()

        for h in head_metrics:
            categories.update(h.scores.keys())

        for category in sorted(categories):
            self.head_plots.plot_category_heatmap(
                head_metrics,
                category=category,
                save_path=(f"{output_dir}/{category}_heatmap.png"),
            )

        self.head_plots.plot_entropy(
            head_metrics,
            save_path=(f"{output_dir}/head_entropy.png"),
        )

        self.head_plots.plot_dominant_category_heatmap(
            head_metrics,
            save_path=(f"{output_dir}/dominant_category_heatmap.png"),
        )

        self.layer_plots.plot_semantic_vs_structural(
            layer_metrics,
            save_path=(f"{output_dir}/semantic_vs_structural.png"),
        )

        self.layer_plots.plot_layer_entropy(
            layer_metrics,
            save_path=(f"{output_dir}/layer_entropy.png"),
        )

        vectors = self.feature_extractor.extract(head_metrics)

        embeddings, _ = self.pca_embedder.fit_transform(vectors)

        clustered = self.cluster_analyzer.cluster(
            vectors=vectors,
            metrics=head_metrics,
            embeddings=embeddings,
            n_clusters=4,
        )

        cluster_summary = self.cluster_summary.summarize(
            clustered,
            head_metrics,
        )

        cluster_summary_str = "\n" + "=" * 60 + "\n"
        cluster_summary_str += "CLUSTER SUMMARY\n"
        cluster_summary_str += "=" * 60 + "\n"
        for cluster, summary in cluster_summary.items():
            cluster_summary_str += (
                f"Cluster {cluster}: "
                f"{summary['dominant_category']} "
                f"(size={summary['size']})\n"
            )
        print(cluster_summary_str)

        self.cluster_plots.plot_pca_clusters(
            clustered,
            save_path=(f"{output_dir}/pca_clusters.png"),
        )

        self.cluster_plots.plot_cluster_distribution_by_layer(
            clustered,
            save_path=(f"{output_dir}/cluster_distribution_by_layer.png"),
        )

        self.cluster_plots.plot_cluster_heatmap(
            clustered,
            save_path=(f"{output_dir}/cluster_heatmap.png"),
        )

        self.cluster_plots.plot_cluster_entropy(
            clustered,
            head_metrics,
            save_path=(f"{output_dir}/cluster_entropy.png"),
        )

        self.cluster_plots.plot_cluster_specialization(
            clustered,
            head_metrics,
            save_path=(f"{output_dir}/cluster_specialization.png"),
        )

        silhouette = self.cluster_validator.silhouette(
            vectors,
            clustered,
        )

        silhouette_str = "\n" + "=" * 60 + "\n"
        silhouette_str += "SILHOUETTE SCORE\n"
        silhouette_str += "=" * 60 + "\n"
        silhouette_str += f"{silhouette:.4f}\n"
        print(silhouette_str)

        divergences = self.cluster_divergence.pairwise_js_divergence(
            head_metrics,
            clustered,
        )

        divergence_str = "\n" + "=" * 60 + "\n"
        divergence_str += "JS DIVERGENCES (Pairwise)\n"
        divergence_str += "=" * 60 + "\n"
        for pair, div in divergences.items():
            divergence_str += f"{pair}: {div:.4f}\n"
        print(divergence_str)

        entropy_stats = self.stats_analyzer.entropy_anova(
            head_metrics,
            clustered,
        )

        entropy_anova_str = "\n" + "=" * 60 + "\n"
        entropy_anova_str += "ENTROPY ANOVA\n"
        entropy_anova_str += "=" * 60 + "\n"
        entropy_anova_str += f"{entropy_stats}\n"
        print(entropy_anova_str)

        specialization_stats = self.stats_analyzer.specialization_anova(
            head_metrics,
            clustered,
        )

        specialization_anova_str = "\n" + "=" * 60 + "\n"
        specialization_anova_str += "SPECIALIZATION ANOVA\n"
        specialization_anova_str += "=" * 60 + "\n"
        specialization_anova_str += f"{specialization_stats}\n"
        print(specialization_anova_str)

        perm = self.permutation_tester.entropy_trend_test(
            layer_metrics,
        )

        perm_str = "\n" + "=" * 60 + "\n"
        perm_str += "ENTROPY TREND PERMUTATION TEST\n"
        perm_str += "=" * 60 + "\n"
        perm_str += f"{perm}\n"
        print(perm_str)

        # Combine all stats and save to file
        all_stats = (
            cluster_summary_str
            + silhouette_str
            + divergence_str
            + entropy_anova_str
            + specialization_anova_str
            + perm_str
        )
        self._save_stats_to_file(output_dir, all_stats)
