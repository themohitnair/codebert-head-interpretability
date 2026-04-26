import os
from dotenv import load_dotenv
from codebert_head_interpretability.datasets import get_dataset
from codebert_head_interpretability.models.codebert import CodeBertModel
from codebert_head_interpretability.pipelines.head_analysis_pipeline import (
    CodeOnlyPipeline,
    CodeQueryPipeline,
    MismatchPipeline,
)

load_dotenv()


LANGUAGE = os.getenv("PROGRAMMING_LANGUAGE", "python")
NUM_EXAMPLES = int(os.getenv("NUM_EXAMPLES", "100"))
DATASET_NAME = os.getenv("DATASET_NAME", "codesearchnet")

HEAD_ANALYSIS_OUTPUT_DIR = "head_analysis_" + LANGUAGE + "_outputs"


def main():

    dataset = get_dataset(DATASET_NAME, language=LANGUAGE)
    codebert_model = CodeBertModel()

    print(
        f"Running head analysis pipelines for {LANGUAGE} with dataset '{DATASET_NAME}' and {NUM_EXAMPLES} examples...\n"
    )

    print("Head Analysis Pipeline - Code Only")
    pipeline = CodeOnlyPipeline(
        dataset,
        codebert_model,
    )
    pipeline.run(
        max_examples=NUM_EXAMPLES, output_dir=HEAD_ANALYSIS_OUTPUT_DIR + "/code_only"
    )

    print("Head Analysis Pipeline - Code Query")
    pipeline = CodeQueryPipeline(
        dataset,
        codebert_model,
    )
    pipeline.run(
        max_examples=NUM_EXAMPLES, output_dir=HEAD_ANALYSIS_OUTPUT_DIR + "/code_query"
    )

    print("Head Analysis Pipeline - Mismatched Code Query")
    pipeline = MismatchPipeline(
        dataset,
        codebert_model,
    )
    pipeline.run(
        max_examples=NUM_EXAMPLES, output_dir=HEAD_ANALYSIS_OUTPUT_DIR + "/mismatch"
    )


if __name__ == "__main__":
    main()
