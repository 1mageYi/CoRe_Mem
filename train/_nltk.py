def ensure_nltk_punkt() -> None:
    """vec2text metrics use NLTK tokenizers; download once if missing."""
    try:
        import nltk

        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)
    except ImportError:
        pass
