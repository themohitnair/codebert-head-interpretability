def get_language_spec(language: str):
    if language == "python":
        from .python_spec import PythonSpec

        return PythonSpec()

    raise ValueError(f"Unknown language: {language}")
