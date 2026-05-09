import argparse
from codebert_head_interpretability.datasets import get_dataset
from codebert_head_interpretability.models.codebert import CodeBertModel
from codebert_head_interpretability.pipelines.head_analysis_pipeline import (
    CodeOnlyPipeline,
    CodeQueryPipeline,
    MismatchPipeline,
)

PIPELINES = {
    "code_only": CodeOnlyPipeline,
    "code_query": CodeQueryPipeline,
    "mismatch": MismatchPipeline,
}


def main():
    parser = argparse.ArgumentParser(
        description="Run head analysis pipelines for CodeBERT interpretability."
    )
    parser.add_argument(
        "--language",
        type=str,
        default="python",
        help="Programming language to analyze (default: python)",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=100,
        help="Number of examples to process (default: 100)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="codesearchnet",
        help="Dataset name to use (default: codesearchnet)",
    )
    parser.add_argument(
        "--pipelines",
        type=str,
        nargs="+",
        choices=["code_only", "code_query", "mismatch"],
        default=["code_only", "code_query", "mismatch"],
        help="Pipelines to run. Options: code_only, code_query, mismatch. Can specify multiple. (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory base (default: head_analysis_<language>_outputs)",
    )

    args = parser.parse_args()

    output_dir = args.output_dir or f"head_analysis_{args.language}_outputs"

    dataset = get_dataset(args.dataset, language=args.language)
    codebert_model = CodeBertModel()

    print(
        f"Running head analysis pipelines for {args.language} with dataset '{args.dataset}' and {args.num_examples} examples...\n"
    )

    for pipeline_name in args.pipelines:
        print(f"Head Analysis Pipeline - {pipeline_name.replace('_', ' ').title()}")
        pipeline_class = PIPELINES[pipeline_name]
        pipeline = pipeline_class(dataset, codebert_model)
        pipeline.run(
            max_examples=args.num_examples,
            output_dir=f"{output_dir}/{pipeline_name}",
        )
        print()


if __name__ == "__main__":
    main()
