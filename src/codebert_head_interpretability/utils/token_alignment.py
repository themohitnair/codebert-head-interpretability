from collections import Counter

from codebert_head_interpretability.schemas.model_output import (
    ModelOutput,
    ModelOutputWithQuery,
    ModelToken,
)

from codebert_head_interpretability.schemas.tokens import (
    AlignedToken,
    ClassifiedToken,
)


def spans_overlap(
    a_start: int,
    a_end: int,
    b_start: int,
    b_end: int,
) -> bool:
    return a_start < b_end and b_start < a_end


def overlap_length(
    a_start: int,
    a_end: int,
    b_start: int,
    b_end: int,
) -> int:
    return max(
        0,
        min(a_end, b_end) - max(a_start, b_start),
    )


def alignment_confidence(
    overlap: int,
    token_length: int,
) -> float:
    if token_length <= 0:
        return 0.0

    return overlap / token_length


def align_window_tokens(
    ast_tokens: list[ClassifiedToken],
    model_tokens: list[ModelToken],
    min_confidence: float = 0.5,
) -> list[AlignedToken]:

    aligned_tokens: list[AlignedToken] = []

    sorted_ast_tokens = sorted(
        ast_tokens,
        key=lambda t: t.start,
    )

    for mt in model_tokens:
        if mt.start == mt.end:
            continue

        best_category = None
        best_overlap = -1

        model_token_length = mt.end - mt.start

        for at in sorted_ast_tokens:
            if at.start >= mt.end:
                break

            if at.end <= mt.start:
                continue

            if not spans_overlap(
                mt.start,
                mt.end,
                at.start,
                at.end,
            ):
                continue

            overlap = overlap_length(
                mt.start,
                mt.end,
                at.start,
                at.end,
            )

            if overlap > best_overlap:
                best_overlap = overlap
                best_category = at.category

        confidence = alignment_confidence(
            best_overlap,
            model_token_length,
        )

        if best_category is None or confidence < min_confidence:
            category = "unknown"

        else:
            category = best_category

        aligned_tokens.append(
            AlignedToken(
                text=mt.text,
                start=mt.start,
                end=mt.end,
                index=mt.index,
                category=category,
                confidence=confidence,
            )
        )

    return aligned_tokens


def align_model_output(
    ast_tokens: list[ClassifiedToken],
    model_output: (ModelOutput | ModelOutputWithQuery),
    min_confidence: float = 0.5,
) -> list[list[AlignedToken]]:

    all_aligned: list[list[AlignedToken]] = []

    for window in model_output.windows:
        aligned = align_window_tokens(
            ast_tokens=ast_tokens,
            model_tokens=window.tokens,
            min_confidence=min_confidence,
        )

        all_aligned.append(aligned)

    return all_aligned


def compute_alignment_stats(
    aligned_tokens: list[AlignedToken],
):
    category_counter = Counter()

    confidences = []

    for token in aligned_tokens:
        category_counter[token.category] += 1

        confidences.append(token.confidence)

    total = sum(category_counter.values())

    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    return {
        "total_tokens": total,
        "avg_confidence": avg_confidence,
        "category_distribution": dict(category_counter),
    }


def print_alignment_stats(
    aligned_tokens: list[AlignedToken],
):
    stats = compute_alignment_stats(aligned_tokens)

    print("\nAlignment Stats:\n")

    print(f"Total Tokens: {stats['total_tokens']}")

    print(f"Average Confidence: {stats['avg_confidence']:.3f}")

    print()

    distribution = stats["category_distribution"]

    total = stats["total_tokens"]

    for cat, count in sorted(
        distribution.items(),
        key=lambda x: x[1],
        reverse=True,
    ):
        pct = 100 * count / total

        print(f"{cat:<15} {count:<8} {pct:.2f}%")
