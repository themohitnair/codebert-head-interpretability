def get_language_spec(language: str):
    if language == "python":
        from .python_spec import PythonSpec

        return PythonSpec()
    elif language == "java":
        from .java_spec import JavaSpec

        return JavaSpec()

    raise ValueError(f"Unknown language: {language}")
