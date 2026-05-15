from .base import LanguageSpec


class JavaSpec(LanguageSpec):
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
            if not keep_comments and stripped_line.startswith("//"):
                continue
            cleaned_lines.append(line)
        return "\n".join(cleaned_lines)

    KEYWORDS = {
        "abstract",
        "assert",
        "boolean",
        "break",
        "byte",
        "case",
        "catch",
        "char",
        "class",
        "const",
        "continue",
        "default",
        "do",
        "double",
        "else",
        "enum",
        "extends",
        "final",
        "finally",
        "float",
        "for",
        "goto",
        "if",
        "implements",
        "import",
        "instanceof",
        "int",
        "interface",
        "long",
        "native",
        "new",
        "package",
        "private",
        "protected",
        "public",
        "return",
        "short",
        "static",
        "strictfp",
        "super",
        "switch",
        "synchronized",
        "this",
        "throw",
        "throws",
        "transient",
        "try",
        "void",
        "volatile",
        "while",
        "true",
        "false",
        "null",
    }

    OPERATORS = {
        "+",
        "-",
        "*",
        "/",
        "%",
        "++",
        "--",
        "=",
        "+=",
        "-=",
        "*=",
        "/=",
        "%=",
        "==",
        "!=",
        ">",
        "<",
        ">=",
        "<=",
        "&&",
        "||",
        "!",
        "&",
        "|",
        "^",
        "~",
        "<<",
        ">>",
        ">>>",
        "&=",
        "|=",
        "^=",
        "<<=",
        ">>=",
        ">>>=",
        "?",
        ":",
        "->",
        "::",
    }

    BRACKETS = {
        "(",
        ")",
        "{",
        "}",
        "[",
        "]",
    }

    DELIMITERS = {
        ",",
        ".",
        ";",
        "@",
    }
