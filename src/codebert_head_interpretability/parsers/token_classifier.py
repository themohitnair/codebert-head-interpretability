def walk_tree(node, code, tokens):

    if len(node.children) == 0:
        token = code[node.start_byte : node.end_byte]
        tokens.append((token, node.type))

    for child in node.children:
        walk_tree(child, code, tokens)


def extract_tokens(code, root_node):

    tokens = []
    walk_tree(root_node, code, tokens)
    return tokens


def classify_token(token, node_type, spec):

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


def classify_tokens(tokens, spec):

    results = []

    for token, node_type in tokens:
        category = classify_token(token, node_type, spec)
        results.append((token, category))

    return results
