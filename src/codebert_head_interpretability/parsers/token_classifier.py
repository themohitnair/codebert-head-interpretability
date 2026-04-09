from codebert_head_interpretability.languages.base_spec import LanguageSpec
from codebert_head_interpretability.schemas.tokens import ASTToken, ClassifiedToken


def _classify_single_token(token: str, node_type: str, spec: LanguageSpec) -> str:
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


def classify_tokens(
    ast_tokens: list[ASTToken], spec: LanguageSpec
) -> list[ClassifiedToken]:
    results: list[ClassifiedToken] = []

    for ast_token in ast_tokens:
        category = _classify_single_token(ast_token.token, ast_token.node_type, spec)
        results.append(
            ClassifiedToken(
                **ast_token.model_dump(),
                category=category,
            )
        )

    return results
