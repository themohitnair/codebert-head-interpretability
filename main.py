import argparse
from codebert_head_interpretability.datasets import get_dataset
from codebert_head_interpretability.models.codebert import CodeBertModel
from codebert_head_interpretability.pipelines.head_analysis_pipeline import (
    CodeOnlyPipeline,
    CodeQueryPipeline,
    InterventionComparisonPipeline,
    MismatchPipeline,
)

PIPELINES = {
    "code_only": CodeOnlyPipeline,
    "code_query": CodeQueryPipeline,
    "intervention_comparison": InterventionComparisonPipeline,
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
        choices=["code_only", "code_query", "intervention_comparison", "mismatch"],
        default=["code_only", "code_query", "intervention_comparison", "mismatch"],
        help="Pipelines to run. Options: code_only, code_query, intervention_comparison, mismatch. Can specify multiple. (default: all)",
    )
    parser.add_argument(
        "--intervention-layer",
        type=int,
        default=0,
        help="Layer index to ablate for intervention_comparison pipeline (default: 0)",
    )
    parser.add_argument(
        "--intervention-head",
        type=int,
        default=0,
        help="Head index to ablate for intervention_comparison pipeline (default: 0)",
    )
    parser.add_argument(
        "--intervention-debug-shapes",
        action="store_true",
        help="Print intervention hook shape diagnostics during inference (intervention_comparison only)",
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
        if pipeline_name == "intervention_comparison":
            pipeline = pipeline_class(
                dataset,
                codebert_model,
                target_layer=args.intervention_layer,
                target_head=args.intervention_head,
                debug_shapes=args.intervention_debug_shapes,
            )
        else:
            pipeline = pipeline_class(dataset, codebert_model)
        pipeline.run(
            max_examples=args.num_examples,
            output_dir=f"{output_dir}/{pipeline_name}",
        )
        print()


if __name__ == "__main__":
    main()
