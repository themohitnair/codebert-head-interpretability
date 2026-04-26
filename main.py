from codebert_head_interpretability.datasets.codesearchnet import CodeSearchNetDataset
from codebert_head_interpretability.models.codebert import CodeBertModel
from codebert_head_interpretability.pipelines.head_analysis_pipeline import (
    CodeOnlyPipeline,
    CodeQueryPipeline,
    MismatchPipeline,
)

LANGUAGE = "python"
NUM_EXAMPLES = 100


def main():

    codesearchnet_dataset = CodeSearchNetDataset(language=LANGUAGE)
    codebert_model = CodeBertModel()

    print("Head Analysis Pipeline - Code Only")
    pipeline = CodeOnlyPipeline(
        codesearchnet_dataset,
        codebert_model,
    )
    pipeline.run(max_examples=NUM_EXAMPLES, output_dir="outputs_code")

    print("Head Analysis Pipeline - Code Query")
    pipeline = CodeQueryPipeline(
        codesearchnet_dataset,
        codebert_model,
    )
    pipeline.run(max_examples=NUM_EXAMPLES, output_dir="outputs_query")

    print("Head Analysis Pipeline - Mismatched Code Query")
    pipeline = MismatchPipeline(
        codesearchnet_dataset,
        codebert_model,
    )
    pipeline.run(max_examples=NUM_EXAMPLES, output_dir="outputs_mismatch")


if __name__ == "__main__":
    main()
