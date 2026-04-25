from codebert_head_interpretability.languages.base_spec import LanguageSpec
from codebert_head_interpretability.parsers.tree_sitter_parser import CodeParser
from codebert_head_interpretability.schemas.tokens import ClassifiedToken


class ClassifyTokens:
    def __init__(self, parser: CodeParser):
        self.parser = parser

    def classify_tokens(self, code: str) -> list[ClassifiedToken]:
        root = self.parser.parse(code)
        ast_tokens = self.parser.get_ast_tokens(root, code)
        results: list[ClassifiedToken] = []

        for ast_token in ast_tokens:
            category = self._classify_single_token(
                ast_token.token, ast_token.node_type, self.parser.get_language_spec()
            )
            results.append(
                ClassifiedToken(
                    **ast_token.model_dump(),
                    category=category,
                )
            )

        return results

    def _classify_single_token(
        self, token: str, node_type: str, spec: LanguageSpec
    ) -> str:
        if token in spec.KEYWORDS:
            return "keyword"

        if node_type == "identifier":
            return "identifier"

        if token in spec.OPERATORS:
            return "operator"

        if token in spec.BRACKETS:
            return "bracket"

        if token in spec.DELIMITERS:
            return "delimiter"

        if node_type in ["string", "integer", "float"]:
            return "literal"

        return "other"
