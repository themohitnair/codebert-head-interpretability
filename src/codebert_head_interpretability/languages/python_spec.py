import keyword
from .base_spec import LanguageSpec


class PythonSpec(LanguageSpec):
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
