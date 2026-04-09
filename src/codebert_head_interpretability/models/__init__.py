def get_model(name: str):

    if name == "codebert":
        from .codebert import CodeBertModel

        return CodeBertModel()

    raise ValueError(f"Unknown model: {name}")
