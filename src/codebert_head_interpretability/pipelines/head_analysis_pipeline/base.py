from codebert_head_interpretability.analytics.analysis import HeadAnalysisAnalyzer
from codebert_head_interpretability.analytics.visualization import (
    HeadAnalysisVisualizer,
)
from codebert_head_interpretability.datasets.base import BaseDataset
from codebert_head_interpretability.models.base import BaseModel
from codebert_head_interpretability.parsers.token_classifier import TokenClassifier
from codebert_head_interpretability.parsers.tree_sitter_parser import CodeParser
from codebert_head_interpretability.schemas.analysis import HeadAnalysisResult
from codebert_head_interpretability.schemas.code_query import CodeQueryModel


class BasePipeline:
    def __init__(self, dataset: BaseDataset, model: BaseModel):
        self.dataset = dataset
        self.model = model
        self.parser = CodeParser(language=dataset.language)
        self.analyzer = HeadAnalysisAnalyzer()
        self.visualizer = HeadAnalysisVisualizer()
        self.classifier = TokenClassifier(parser=self.parser)

    def process_example(self, example: CodeQueryModel) -> list[HeadAnalysisResult]:
        raise NotImplementedError

    def run(self, split="train", max_examples=100, output_dir="outputs"):
        ds = self.dataset.load(split)

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

    def _visualize(self, results: list[HeadAnalysisResult], output_dir: str):
        if not results:
            print("No results to visualize.")
            return

        categories = set()
        for r in results:
            categories.update(r.distribution.scores.keys())

        for category in sorted(categories):
            self.visualizer.plot_category_heatmap(
                results,
                category=category,
                save_path=f"{output_dir}/{category}_heatmap.png",
            )

        self.visualizer.plot_top_category_map(
            results,
            save_path=f"{output_dir}/top_category_map.png",
        )

        self.visualizer.plot_head_distribution(
            results,
            save_path=f"{output_dir}/head_distribution.png",
        )

        self.visualizer.plot_entropy(
            results,
            save_path=f"{output_dir}/entropy.png",
        )
