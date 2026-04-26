from codebert_head_interpretability.schemas.analysis import HeadAnalysisResult
from codebert_head_interpretability.utils.token_alignment import align_model_output
from .base import BasePipeline
from codebert_head_interpretability.schemas.code_query import CodeQueryModel


class CodeQueryPipeline(BasePipeline):
    def process_example(self, example: CodeQueryModel) -> list[HeadAnalysisResult]:
        code = example.code
        query = example.query

        classified_tokens = self.classifier.classify_tokens(code)

        output = self.model.run_query_code(query, code)

        aligned_windows = align_model_output(classified_tokens, output)

        results: list[HeadAnalysisResult] = []

        for window, aligned_tokens in zip(output.windows, aligned_windows):
            stats = self.analyzer.analyze_query_to_code(
                aligned_tokens,
                window.attentions,
                window.query_len,
            )
            results.extend(stats)

        return results
