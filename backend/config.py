import os

def _env_bool(name: str, default: bool = False) -> bool:
    """Read a true/false setting. Accepts true, 1, yes, on"""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


class Settings:
    """All configuration lives here. Each value comes from an environment
    variable, with a sensible default if the variable isn't set."""

    def __init__(self):
        self.OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
        self.MODEL_NAME = os.getenv("MODEL_NAME", "llama3.2:1b")
        self.CHROMA_PATH = os.getenv("CHROMA_PATH", "./rag_db")
        self.DOCS_DIR = os.getenv("DOCS_DIR", "docs")
        self.MAX_RESULTS = int(os.getenv("MAX_RESULTS", "3"))
        self.MAX_DISTANCE = float(os.getenv("MAX_DISTANCE", "1.2"))
        # A best-match distance below this counts as "high" confidence
        self.CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
        # Below this (but above the one above) counts as "medium"
        self.MEDIUM_CONFIDENCE_THRESHOLD = float(os.getenv("MEDIUM_CONFIDENCE_THRESHOLD", "1.0"))
        self.DEBUG = _env_bool("DEBUG", False)
        self.TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))

    def as_dict(self) -> dict:
        return dict(vars(self))
    

settings = Settings()