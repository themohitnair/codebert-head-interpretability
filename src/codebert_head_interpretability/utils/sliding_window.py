def create_sliding_windows(
    input_ids: list[int],
    offsets: list[tuple],
    max_length: int,
    stride: int,
):
    windows = []

    start = 0
    n = len(input_ids)

    while start < n:
        end = start + max_length

        window_ids = input_ids[start:end]
        window_offsets = offsets[start:end]

        windows.append((window_ids, window_offsets))

        if end >= n:
            break

        start += stride

    return windows


def build_query_code_window(
    tokenizer,
    query_ids,
    code_ids,
    code_offsets,
    max_length,
    stride,
):
    # TODO: check how the query to code ratio affects the results

    # reserve space for special tokens (3 special tokens: <s>, </s>, </s> for query and code separation)
    if len(query_ids) + 3 >= max_length:
        query_ids = query_ids[: max_length - 3]

    query_len = len(query_ids)

    reserved = query_len + 3
    code_max_len = max_length - reserved

    windows = create_sliding_windows(
        code_ids,
        code_offsets,
        code_max_len,
        stride,
    )

    combined_windows = []

    for code_chunk_ids, code_chunk_offsets in windows:
        input_ids = (
            [tokenizer.cls_token_id]
            + query_ids
            + [tokenizer.sep_token_id]
            + code_chunk_ids
            + [tokenizer.sep_token_id]
        )

        attention_mask = [1] * len(input_ids)

        combined_windows.append(
            (
                input_ids,
                attention_mask,
                code_chunk_offsets,
                query_len,
            )
        )

    return combined_windows
