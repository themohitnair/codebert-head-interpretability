from codebert_head_interpretability.parsers import (
    CodeParser,
    extract_tokens,
    classify_tokens,
)
from codebert_head_interpretability.languages.python_spec import PythonSpec


def main():

    code = """
def add(a, b):
    return a + b
"""

    parser = CodeParser()

    root = parser.parse(code)

    tokens = extract_tokens(code, root)

    # node_type from tree-sitter does not classify it properly, so we need to classify it manually
    classified = classify_tokens(tokens, PythonSpec())

    for token, category in classified:
        print(token, category)


if __name__ == "__main__":
    main()
