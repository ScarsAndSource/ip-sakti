import os
import sys

from dotenv import load_dotenv

load_dotenv()

_REQUIRED = {
    "DATABASE_URL": "Supabase connection string (Session mode pooler recommended) - see README setup step 2.",
    "GROQ_API_KEY": "Free API key from console.groq.com - see README setup step 2.",
}


def _require(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        hint = _REQUIRED.get(name, "")
        print(f"FATAL: missing required environment variable {name}. {hint}", file=sys.stderr)
        print("Copy .env.example to .env and fill it in before starting the server.", file=sys.stderr)
        raise SystemExit(1)
    return value


class Settings:
    DATABASE_URL: str = _require("DATABASE_URL")
    GROQ_API_KEY: str = _require("GROQ_API_KEY")
    EMBEDDING_MODEL: str = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    RETRIEVAL_TOP_K: int = int(os.environ.get("RETRIEVAL_TOP_K", 8))
    CONFIDENCE_THRESHOLD: float = float(os.environ.get("CONFIDENCE_THRESHOLD", 0.55))

    # --- DB pool / pgvector ---
    DB_POOL_MIN_SIZE: int = int(os.environ.get("DB_POOL_MIN_SIZE", 2))
    DB_POOL_MAX_SIZE: int = int(os.environ.get("DB_POOL_MAX_SIZE", 10))
    DB_COMMAND_TIMEOUT_SECONDS: float = float(os.environ.get("DB_COMMAND_TIMEOUT_SECONDS", 10))
    PGVECTOR_PROBES: int = int(os.environ.get("PGVECTOR_PROBES", 10))

    # --- Retrieval ---
    HYBRID_SEARCH_ENABLED: bool = os.environ.get("HYBRID_SEARCH_ENABLED", "true").lower() == "true"
    # A fresh Render/Supabase deployment otherwise has an empty `chunks`
    # table and every query abstains before reaching Groq.
    AUTO_BOOTSTRAP_CORPUS: bool = os.environ.get("AUTO_BOOTSTRAP_CORPUS", "true").lower() == "true"

    # --- Caching ---
    CACHE_TTL_SECONDS: float = float(os.environ.get("CACHE_TTL_SECONDS", 600))
    CACHE_MAX_SIZE: int = int(os.environ.get("CACHE_MAX_SIZE", 500))

    # --- Concurrency ---
    EMBED_CONCURRENCY_LIMIT: int = int(os.environ.get("EMBED_CONCURRENCY_LIMIT", 4))

    # --- Groq fallback ladder ---
    GROQ_MODEL: str = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
    GROQ_FALLBACK_MODEL: str = os.environ.get("GROQ_FALLBACK_MODEL", "llama-3.1-8b-instant")
    GROQ_MAX_RETRIES: int = int(os.environ.get("GROQ_MAX_RETRIES", 2))
    GROQ_RETRY_BACKOFF_SECONDS: float = float(os.environ.get("GROQ_RETRY_BACKOFF_SECONDS", 0.5))
    GROQ_TIMEOUT_SECONDS: float = float(os.environ.get("GROQ_TIMEOUT_SECONDS", 15))

    # --- Request-level protection ---
    # The Groq ladder can make two primary attempts and one fallback attempt.
    # Its own default timeouts total 45.5s including retry backoff, so a 20s
    # route timeout cancelled real generation before the fallback could run.
    QUERY_TIMEOUT_SECONDS: float = float(os.environ.get("QUERY_TIMEOUT_SECONDS", 55))
    RATE_LIMIT_PER_MINUTE: int = int(os.environ.get("RATE_LIMIT_PER_MINUTE", 30))


settings = Settings()
