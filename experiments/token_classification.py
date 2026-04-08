from codebert_head_interpretability.dataset import get_dataset
from codebert_head_interpretability.parsers.token_classifier import classify_tokens
from codebert_head_interpretability.parsers.tree_sitter_parser import CodeParser
from codebert_head_interpretability.languages.python_spec import PythonSpec


def main():
    """
    CODEBERT HEAD INTERPRETABILITY

    step 1: get code query pairs from the dataset [x]
    step 2: parse code into ast and store all the ast nodes [x]
    step 3: classify every nodes [x]
    step 4: Tokenize using codebert tokenizer []
    step 5: Token alignment between bpe and ast nodes []
    step 6: extract attention weights from codebert and analyze them (code only) []
    step 7: extract attention weights from codebert and analyze them (code + query) []
    """

    dataset = get_dataset("codesearchnet", language="python")
    ds = dataset.load(split="train")
    parser = CodeParser(language="python")

    for example in dataset.to_examples(ds, max_examples=1):
        root_node = parser.parse(example.code)
        ast_tokens = parser.get_ast_tokens(root_node, example.code)
        classified_tokens = classify_tokens(ast_tokens, PythonSpec())
        for token in classified_tokens:
            print(
                f"Token: {token.token}, Type: {token.node_type}, Category: {token.category}"
            )


if __name__ == "__main__":
    main()
