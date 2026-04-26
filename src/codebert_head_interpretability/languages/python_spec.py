import keyword
from .base import LanguageSpec


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
