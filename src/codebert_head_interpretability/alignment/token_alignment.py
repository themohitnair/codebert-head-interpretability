from codebert_head_interpretability.schemas.tokens import (
    ClassifiedToken,
    AlignedToken,
)
from codebert_head_interpretability.schemas.model_output import (
    ModelOutput,
    ModelToken,
)


def spans_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start < b_end and b_start < a_end


def align_window_tokens(
    ast_tokens: list[ClassifiedToken],
    model_tokens: list[ModelToken],
) -> list[AlignedToken]:

    aligned_tokens: list[AlignedToken] = []

    for mt in model_tokens:
        # skip special tokens (offsets 0, 0)
        if mt.start == mt.end:
            continue

        matched_categories = []

        # TODO: we have to optimize this
        for at in ast_tokens:
            if spans_overlap(mt.start, mt.end, at.start, at.end):
                matched_categories.append(at.category)

        if not matched_categories:
            category = "other"
        else:
            # majority vote
            category = max(set(matched_categories), key=matched_categories.count)

        aligned_tokens.append(
            AlignedToken(
                text=mt.text,
                start=mt.start,
                end=mt.end,
                index=mt.index,
                category=category,
            )
        )

    return aligned_tokens


def align_model_output(
    ast_tokens: list[ClassifiedToken],
    model_output: ModelOutput,
) -> list[list[AlignedToken]]:

    all_aligned: list[list[AlignedToken]] = []

    for window in model_output.windows:
        aligned = align_window_tokens(
            ast_tokens,
            window.tokens,
        )

        all_aligned.append(aligned)

    return all_aligned
