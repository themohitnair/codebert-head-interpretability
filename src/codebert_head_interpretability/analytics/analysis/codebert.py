from collections import defaultdict

from codebert_head_interpretability.schemas.analysis import (
    CategoryDistribution,
    HeadAnalysisResult,
)
from codebert_head_interpretability.schemas.tokens import AlignedToken
from codebert_head_interpretability.utils.maths import compute_entropy


class HeadAnalysisAnalyzer:
    def _aggregate_and_normalize(
        self, category_scores, category_counts
    ) -> CategoryDistribution:
        for cat in category_scores:
            if category_counts[cat] > 0:
                category_scores[cat] /= category_counts[cat]

        total = sum(category_scores.values())
        if total > 0:
            for cat in category_scores:
                category_scores[cat] /= total

        return CategoryDistribution(scores=dict(category_scores))

    def _analyze_head(
        self, tokens: list[AlignedToken], score_fn
    ) -> tuple[CategoryDistribution, float]:
        category_scores = defaultdict(float)
        category_counts = defaultdict(int)

        for token in tokens:
            if token is None or token.category == "unknown":
                continue

            score = score_fn(token)

            if score is None:
                continue

            cat = token.category

            category_scores[cat] += score
            category_counts[cat] += 1

        distribution = self._aggregate_and_normalize(category_scores, category_counts)

        entropy = compute_entropy(distribution.scores)

        return distribution, entropy

    def analyze_code_only(
        self, aligned_tokens: list[AlignedToken], attentions
    ) -> list[HeadAnalysisResult]:
        results: list[HeadAnalysisResult] = []

        for layer_idx, layer_attn in enumerate(attentions):
            layer_attn = layer_attn.squeeze(0)

            num_heads = layer_attn.shape[0]
            seq_len = layer_attn.shape[1]

            for head_idx in range(num_heads):
                head_attn = layer_attn[head_idx]

                def score_fn(token):
                    idx = token.index
                    if idx >= seq_len:
                        return None
                    return head_attn[:, idx].sum().item()

                distribution, entropy = self._analyze_head(aligned_tokens, score_fn)

                results.append(
                    HeadAnalysisResult(
                        layer=layer_idx,
                        head=head_idx,
                        distribution=distribution,
                        entropy=entropy,
                    )
                )

        return results

    def analyze_query_to_code(
        self, aligned_tokens: list[AlignedToken], attentions, query_len: int
    ) -> list[HeadAnalysisResult]:
        results: list[HeadAnalysisResult] = []

        for layer_idx, layer_attn in enumerate(attentions):
            layer_attn = layer_attn.squeeze(0)

            num_heads = layer_attn.shape[0]

            query_indices = range(1, 1 + query_len)
            code_start = query_len + 2
            code_end = len(aligned_tokens) - 1

            code_tokens = aligned_tokens[code_start:code_end]

            for head_idx in range(num_heads):
                head_attn = layer_attn[head_idx]

                def score_fn(token):
                    c_idx = token.index

                    val = 0.0
                    for q_idx in query_indices:
                        val += head_attn[q_idx, c_idx].item()

                    return val

                distribution, entropy = self._analyze_head(code_tokens, score_fn)

                results.append(
                    HeadAnalysisResult(
                        layer=layer_idx,
                        head=head_idx,
                        distribution=distribution,
                        entropy=entropy,
                    )
                )

        return results
