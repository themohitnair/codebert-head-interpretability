from collections import defaultdict


from codebert_head_interpretability.datasets import get_dataset
from codebert_head_interpretability.parsers.tree_sitter_parser import CodeParser
from codebert_head_interpretability.parsers.token_classifier import classify_tokens
from codebert_head_interpretability.languages.python_spec import PythonSpec
from codebert_head_interpretability.models.codebert import CodeBertModel
from codebert_head_interpretability.alignment.token_alignment import align_model_output
from codebert_head_interpretability.analysis.code_analysis import analyze_attention
from codebert_head_interpretability.analysis.visualization import (
    plot_category_heatmap,
    plot_top_category,
    plot_entropy,
)


def main():

    dataset = get_dataset("codesearchnet", language="python")
    ds = dataset.load(split="train")

    parser = CodeParser(language="python")
    model = CodeBertModel()

    global_stats = defaultdict(lambda: defaultdict(float))
    count_per_head = defaultdict(int)

    for example in dataset.to_examples(ds, max_examples=100):
        code = example.code

        root = parser.parse(code)
        ast_tokens = parser.get_ast_tokens(root, code)
        classified_tokens = classify_tokens(ast_tokens, PythonSpec())

        output = model.run_code(code)

        aligned_windows = align_model_output(classified_tokens, output)

        for window, aligned_tokens in zip(output.windows, aligned_windows):
            stats = analyze_attention(aligned_tokens, window.attentions)

            for entry in stats:
                key = (entry["layer"], entry["head"])

                for cat, val in entry["distribution"].items():
                    global_stats[key][cat] += val

                count_per_head[key] += 1

    # ===================== PRINT =====================

    print("\n=== FINAL HEAD ANALYSIS ===\n")

    for (layer, head), cat_dict in sorted(global_stats.items()):
        total = count_per_head[(layer, head)]

        avg = {k: v / total for k, v in cat_dict.items()}
        top_cat = max(avg, key=avg.get)

        print(f"Layer {layer} | Head {head} | Top: {top_cat}")

        for cat, val in sorted(avg.items(), key=lambda x: -x[1]):
            print(f"  {cat:<12}: {val:.3f}")

        print()

    # ===================== PLOTS =====================

    plot_category_heatmap(
        global_stats,
        count_per_head,
        "identifier",
        save_path="outputs/identifier_heatmap.png",
    )
    plot_category_heatmap(
        global_stats, count_per_head, "keyword", save_path="outputs/keyword_heatmap.png"
    )
    plot_category_heatmap(
        global_stats,
        count_per_head,
        "delimiter",
        save_path="outputs/delimiter_heatmap.png",
    )

    plot_top_category(
        global_stats, count_per_head, save_path="outputs/top_category.png"
    )
    plot_entropy(global_stats, count_per_head, save_path="outputs/entropy.png")


if __name__ == "__main__":
    main()
