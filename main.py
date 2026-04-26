from codebert_head_interpretability.datasets.codesearchnet import CodeSearchNetDataset
from codebert_head_interpretability.models.codebert import CodeBertModel
from codebert_head_interpretability.pipelines.head_analysis_pipeline.code_only import (
    CodeOnlyPipeline,
)
from codebert_head_interpretability.pipelines.head_analysis_pipeline.code_query import (
    CodeQueryPipeline,
)

LANGUAGE = "python"
NUM_EXAMPLES = 10


def main():

    codesearchnet_dataset = CodeSearchNetDataset(language=LANGUAGE)
    codebert_model = CodeBertModel()
    pipeline = CodeOnlyPipeline(
        codesearchnet_dataset,
        codebert_model,
    )
    pipeline.run(max_examples=NUM_EXAMPLES, output_dir="outputs")

    pipeline = CodeQueryPipeline(
        codesearchnet_dataset,
        codebert_model,
    )
    pipeline.run(max_examples=NUM_EXAMPLES, output_dir="outputs_query")


if __name__ == "__main__":
    main()
