from collections import defaultdict

from codebert_head_interpretability.utils.maths import compute_entropy


def analyze_attention(aligned_tokens, attentions):

    layer_head_stats = []

    for layer_idx, layer_attn in enumerate(attentions):
        layer_attn = layer_attn.squeeze(0)

        num_heads = layer_attn.shape[0]
        seq_len = layer_attn.shape[1]

        for head_idx in range(num_heads):
            head_attn = layer_attn[head_idx]

            category_scores = defaultdict(float)
            category_counts = defaultdict(int)

            for token in aligned_tokens:
                idx = token.index

                if idx >= seq_len:
                    continue

                score = head_attn[:, idx].sum().item()

                category_scores[token.category] += score
                category_counts[token.category] += 1

            for cat in category_scores:
                if category_counts[cat] > 0:
                    category_scores[cat] /= category_counts[cat]

            total = sum(category_scores.values())
            if total > 0:
                for cat in category_scores:
                    category_scores[cat] /= total

            entropy = compute_entropy(category_scores)

            layer_head_stats.append(
                {
                    "layer": layer_idx,
                    "head": head_idx,
                    "distribution": dict(category_scores),
                    "entropy": entropy,
                }
            )

    return layer_head_stats
