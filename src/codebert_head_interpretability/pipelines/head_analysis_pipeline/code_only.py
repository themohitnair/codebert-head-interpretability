from codebert_head_interpretability.schemas.analysis import HeadAnalysisResult
from codebert_head_interpretability.schemas.code_query import CodeQueryModel
from codebert_head_interpretability.utils.token_alignment import align_model_output
from .base import BasePipeline


class CodeOnlyPipeline(BasePipeline):
    def process_example(self, example: CodeQueryModel) -> list[HeadAnalysisResult]:
        code = example.code

        classified_tokens = self.classifier.classify_tokens(code)

        output = self.model.run_code(code)

        aligned_windows = align_model_output(classified_tokens, output)

        results: list[HeadAnalysisResult] = []

        for window, aligned_tokens in zip(output.windows, aligned_windows):
            stats = self.analyzer.analyze_code_only(
                aligned_tokens,
                window.attentions,
            )
            results.extend(stats)

        return results
