from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Gemini API
    GEMINI_API_KEY: str = ""

    # Qdrant (optional - for vector RAG)
    QDRANT_URL: str = ""
    QDRANT_API_KEY: str = ""

    # Auth
    ADMIN_API_KEY: str = "admin-secret"
    RAG_API_KEYS: str = "dev-key-1"

    # Gemini model selection (free tier options)
    GEMINI_MODEL: str = "gemini-2.0-flash"

    @property
    def api_keys(self) -> List[str]:
        return [k.strip() for k in self.RAG_API_KEYS.split(",") if k.strip()]

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
