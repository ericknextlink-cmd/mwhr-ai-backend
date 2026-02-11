from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    PROJECT_NAME: str = "MWHWR AI Service"
    API_V1_STR: str = "/api/v1"
    
    # AI Keys
    OPENAI_API_KEY: str
    UNSTRUCTURED_API_KEY: Optional[str] = None
    UNSTRUCTURED_API_URL: str = "https://api.unstructured.io"
    
    # Security (Basic protection for the microservice)
    SERVICE_API_KEY: str = "change_this_to_secure_key"

    class Config:
        case_sensitive = True
        env_file = ".env"

settings = Settings()
