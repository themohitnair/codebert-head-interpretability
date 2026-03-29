from codebert_head_interpretability.parsers import (
    CodeParser,
    extract_tokens,
    classify_tokens,
)
from codebert_head_interpretability.languages.python_spec import PythonSpec
from codebert_head_interpretability.dataset import get_dataset


def main():

    dataset = get_dataset("codesearchnet", language="python")
    ds = dataset.load(split="train")

    parser = CodeParser()

    for example in dataset.to_examples(ds):
        code = example.code
        root = parser.parse(code)

        tokens = extract_tokens(code, root)

        # node_type from tree-sitter does not classify it properly, so we need to classify it manually
        classified = classify_tokens(tokens, PythonSpec())

        for token, category in classified:
            print(token, category)
        break


if __name__ == "__main__":
    main()
