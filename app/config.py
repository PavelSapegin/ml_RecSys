from pathlib import Path

from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):

    MODEL_DIR: Path = BASE_DIR / "models"

    @property
    def MODEL_PATH(self) -> Path:
        return self.MODEL_DIR / "hybrid_model.joblib"

    class Config:
        env_file = ".env"


config = Settings()    
