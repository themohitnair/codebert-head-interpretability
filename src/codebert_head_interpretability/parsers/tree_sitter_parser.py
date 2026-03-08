import tree_sitter_languages
from tree_sitter import Parser


class CodeParser:
    def __init__(self, language="python"):
        lang = tree_sitter_languages.get_language(language)
        self.parser = Parser()
        self.parser.set_language(lang)

    def parse(self, code):
        tree = self.parser.parse(bytes(code, "utf8"))
        return tree.root_node
