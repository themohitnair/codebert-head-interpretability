from codebert_head_interpretability.datasets import get_dataset
from codebert_head_interpretability.parsers.token_classifier import classify_tokens
from codebert_head_interpretability.parsers.tree_sitter_parser import CodeParser
from codebert_head_interpretability.languages.python_spec import PythonSpec
from codebert_head_interpretability.models.codebert import CodeBertModel


def main():
    """
    CODEBERT HEAD INTERPRETABILITY

    step 1: get code query pairs from the dataset [x]
    step 2: parse code into ast and store all the ast nodes [x]
    step 3: classify every nodes [x]
    step 4: Tokenize using codebert tokenizer [x]
    step 5: Token alignment between bpe and ast nodes []
    step 6: extract attention weights from codebert and analyze them (code only) []
    step 7: extract attention weights from codebert and analyze them (code + query) []
    """

    dataset = get_dataset("codesearchnet", language="python")
    ds = dataset.load(split="train")

    parser = CodeParser(language="python")
    model = CodeBertModel()

    for example in dataset.to_examples(ds):
        code = example.code

        print("\n" + "=" * 80)
        print("CODE:\n", code)
        print("=" * 80)

        root_node = parser.parse(code)
        ast_tokens = parser.get_ast_tokens(root_node, code)
        classified_tokens = classify_tokens(ast_tokens, PythonSpec())

        print("\n--- AST TOKENS ---")
        for token in classified_tokens:
            print(
                f"{token.token:<15} | {token.category:<12} | ({token.start}, {token.end})"
            )

        output = model.run_code(code)

        print("\n--- BPE TOKENS ---")
        for window in output.windows:
            for tok in window.tokens:
                print(f"{tok.text:<15} | ({tok.start}, {tok.end}) | idx={tok.index}")

            print("\n--- ATTENTION SHAPE ---")
            print(f"Layers: {len(window.attentions)}")
            print(f"Heads per layer: {len(window.attentions[0])}")
            print(f"Token count: {len(window.tokens)}")

        break


if __name__ == "__main__":
    main()
