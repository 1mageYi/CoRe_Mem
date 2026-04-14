from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_inversion_dir() -> Path:
    return repo_root() / "outputs" / "vec2text" / "minilm" / "inversion"


def default_corrector_dir() -> Path:
    return repo_root() / "outputs" / "vec2text" / "minilm" / "corrector"


def default_vec2text_cache_dir() -> Path:
    """Same default as vec2text.experiments DATASET_CACHE_PATH when env is unset."""
    return Path.home() / ".cache" / "inversion"
