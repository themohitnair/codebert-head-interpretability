from collections import defaultdict

from codebert_head_interpretability.analytics.analysis import AttentionAnalyzer
from codebert_head_interpretability.datasets import get_dataset
from codebert_head_interpretability.parsers import (
    CodeParser,
    ClassifyTokens,
)
from codebert_head_interpretability.models.codebert import CodeBertModel
from codebert_head_interpretability.utils.token_alignment import align_model_output
from codebert_head_interpretability.analytics.visualization import (
    AttentionVisualizer,
)

NUM_EXAMPLES = 10
OUTPUT_DIR = "outputs_query"


def main():
    dataset = get_dataset("codesearchnet", language="python")
    ds = dataset.load(split="train")

    parser = CodeParser(language="python")
    model = CodeBertModel()
    analyzer = AttentionAnalyzer()
    visualizer = AttentionVisualizer()
    classifier = ClassifyTokens(parser=parser)

    global_stats = defaultdict(lambda: defaultdict(float))
    count_per_head = defaultdict(int)

    print("\nProcessing query → code attention...\n")

    for i, example in enumerate(dataset.to_examples(ds, max_examples=NUM_EXAMPLES)):
        code = example.code
        query = example.query

        try:
            # NOTE: Classification and model inference could be parallelized but might take too much memory
            classified_tokens = classifier.classify_tokens(code)

            output = model.run_query_code(query, code)

            aligned_windows = align_model_output(classified_tokens, output)

            for window, aligned_tokens in zip(output.windows, aligned_windows):
                stats = analyzer.analyze_query_to_code(
                    aligned_tokens,
                    window.attentions,
                    window.query_len,
                )

                for entry in stats:
                    key = (entry["layer"], entry["head"])

                    for cat, val in entry["distribution"].items():
                        global_stats[key][cat] += val

                    count_per_head[key] += 1

        except Exception as e:
            print(f"Skipping example {i}: {e}")
            continue

        if i % 10 == 0:
            print(f"Processed {i} examples...")

    # ===================== PRINT =====================

    print("\n=== QUERY → CODE HEAD ANALYSIS ===\n")

    for (layer, head), cat_dict in sorted(global_stats.items()):
        total = count_per_head[(layer, head)]

        avg = {k: v / total for k, v in cat_dict.items()}
        top_cat = max(avg, key=avg.get)

        print(f"Layer {layer} | Head {head} | Top: {top_cat}")

        for cat, val in sorted(avg.items(), key=lambda x: -x[1]):
            print(f"  {cat:<12}: {val:.3f}")

        print()

    # ===================== VISUALIZATIONS =====================

    print("\nGenerating visualizations...\n")

    categories = set()
    for cat_dict in global_stats.values():
        categories.update(cat_dict.keys())

    for category in categories:
        visualizer.plot_category_heatmap(
            global_stats,
            count_per_head,
            category=category,
            save_path=f"{OUTPUT_DIR}/{category}_heatmap.png",
        )

    visualizer.plot_top_category_map(
        global_stats,
        count_per_head,
        save_path=f"{OUTPUT_DIR}/top_category_map.png",
    )

    visualizer.plot_head_distribution(
        global_stats,
        count_per_head,
        save_path=f"{OUTPUT_DIR}/head_distribution.png",
    )

    print(f"\nSaved outputs to '{OUTPUT_DIR}/'\n")


if __name__ == "__main__":
    main()
