import keyword
from .base import LanguageSpec


class PythonSpec(LanguageSpec):
    @classmethod
    def cleanup_code(cls, code: str, keep_comments: bool = False) -> str:
        """Remove comments and docstrings from the code."""
        lines = code.splitlines()
        cleaned_lines = []
        in_docstring = False
        for line in lines:
            stripped_line = line.strip()
            if stripped_line.startswith('"""') or stripped_line.startswith("'''"):
                in_docstring = not in_docstring
                continue
            if in_docstring:
                continue
            if not keep_comments and stripped_line.startswith("#"):
                continue
            cleaned_lines.append(line)
        return "\n".join(cleaned_lines)

    KEYWORDS = set(keyword.kwlist)

    OPERATORS = {
        "+",
        "-",
        "*",
        "/",
        "%",
        "**",
        "=",
        "+=",
        "-=",
        "*=",
        "/=",
        "%=",
        "**=",
        "==",
        "!=",
        ">",
        "<",
        ">=",
        "<=",
        "&",
        "|",
        "^",
        "~",
        "<<",
        ">>",
    }

    BRACKETS = {"(", ")", "{", "}", "[", "]"}

    DELIMITERS = {",", ":", ".", ";"}
