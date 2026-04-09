import tree_sitter_languages
from tree_sitter import Parser

from codebert_head_interpretability.schemas.tokens import ASTToken


class CodeParser:
    def __init__(self, language="python"):
        lang = tree_sitter_languages.get_language(language)
        self.parser = Parser()
        self.parser.set_language(lang)

    def parse(self, code):
        tree = self.parser.parse(bytes(code, "utf8"))
        return tree.root_node

    def get_ast_tokens(self, node, code) -> list[ASTToken]:
        tokens: list[ASTToken] = []
        self._walk(node, tokens, code)
        return tokens

    def _walk(self, node, tokens, code) -> None:
        if len(node.children) == 0:
            token = ASTToken(
                token=code[node.start_byte : node.end_byte],
                start=node.start_byte,
                end=node.end_byte,
                node_type=node.type,
            )
            tokens.append(token)
        for child in node.children:
            self._walk(child, tokens, code)
