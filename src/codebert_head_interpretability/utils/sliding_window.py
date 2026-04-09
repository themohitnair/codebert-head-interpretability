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
    # reserve space for special tokens (3 special tokens: <s>, </s>, </s> for query and code separation)
    reserved = len(query_ids) + 3
    code_max_len = max_length - reserved

    windows = create_sliding_windows(code_ids, code_offsets, code_max_len, stride)

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

        combined_windows.append((input_ids, attention_mask, code_chunk_offsets))

    return combined_windows
