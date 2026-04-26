import random
from typing import Iterator

from codebert_head_interpretability.schemas.code_query import CodeQueryModel


def generate_mismatched_examples(
    examples: list[CodeQueryModel],
) -> Iterator[CodeQueryModel]:
    examples_list = list(examples)

    queries = [ex.query for ex in examples_list]
    codes = [ex.code for ex in examples_list]

    shuffled_queries = queries[:]

    while True:
        random.shuffle(shuffled_queries)

        if all(q1 != q2 for q1, q2 in zip(queries, shuffled_queries)):
            break

    for code, query in zip(codes, shuffled_queries):
        yield CodeQueryModel(code=code, query=query)


def main():
    pass


if __name__ == "__main__":
    main()
