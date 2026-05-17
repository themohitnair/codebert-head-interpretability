from codebert_head_interpretability.interventions import HeadAblationIntervention
from codebert_head_interpretability.schemas.analysis import (
    CategoryDistribution,
    HeadAnalysisResult,
)
from codebert_head_interpretability.schemas.code_query import CodeQueryModel
from codebert_head_interpretability.utils.token_alignment import align_model_output
from .base import BasePipeline


class InterventionComparisonPipeline(BasePipeline):
    def __init__(
        self,
        dataset,
        model,
        target_layer: int = 0,
        target_head: int = 0,
        debug_shapes: bool = False,
    ):
        super().__init__(dataset, model)
        self.target_layer = target_layer
        self.target_head = target_head
        self.debug_shapes = debug_shapes

    def process_example(self, example: CodeQueryModel) -> list[HeadAnalysisResult]:
        code = example.code
        query = example.query

        classified_tokens = self.classifier.classify_tokens(code)

        baseline_output = self.model.run_query_code(query, code, intervention=None)

        ablation = HeadAblationIntervention(
            layer_idx=self.target_layer,
            head_idx=self.target_head,
            debug_shapes=self.debug_shapes,
        )
        intervened_output = self.model.run_query_code(
            query,
            code,
            intervention=ablation,
        )

        baseline_aligned = align_model_output(classified_tokens, baseline_output)
        intervened_aligned = align_model_output(classified_tokens, intervened_output)

        baseline_stats: list[HeadAnalysisResult] = []
        intervened_stats: list[HeadAnalysisResult] = []

        for window, aligned_tokens in zip(baseline_output.windows, baseline_aligned):
            stats = self.analyzer.analyze_query_to_code(
                aligned_tokens,
                window.attentions,
                window.query_len,
            )
            baseline_stats.extend(stats)

        for window, aligned_tokens in zip(intervened_output.windows, intervened_aligned):
            stats = self.analyzer.analyze_query_to_code(
                aligned_tokens,
                window.attentions,
                window.query_len,
            )
            intervened_stats.extend(stats)

        return self._compute_deltas(baseline_stats, intervened_stats)

    def _compute_deltas(
        self,
        baseline_stats: list[HeadAnalysisResult],
        intervened_stats: list[HeadAnalysisResult],
    ) -> list[HeadAnalysisResult]:
        baseline_map = {(r.layer, r.head): r for r in baseline_stats}
        intervened_map = {(r.layer, r.head): r for r in intervened_stats}

        keys = sorted(set(baseline_map.keys()) & set(intervened_map.keys()))
        deltas: list[HeadAnalysisResult] = []

        for layer, head in keys:
            base = baseline_map[(layer, head)]
            intv = intervened_map[(layer, head)]

            categories = set(base.distribution.scores.keys()) | set(
                intv.distribution.scores.keys()
            )

            score_delta = {
                category: intv.distribution.scores.get(category, 0.0)
                - base.distribution.scores.get(category, 0.0)
                for category in categories
            }

            deltas.append(
                HeadAnalysisResult(
                    layer=layer,
                    head=head,
                    distribution=CategoryDistribution(scores=score_delta),
                    entropy=(intv.entropy - base.entropy),
                )
            )

        return deltas
