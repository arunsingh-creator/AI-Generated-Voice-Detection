"""
Configuration management for the AI Voice Detection API
"""
import os
from typing import List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # API Configuration
    API_KEYS: str = "test_api_key_123"
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"
    ENVIRONMENT: str = "development"
    
    # Model Configuration
    MODEL_PATH: str = "models/voice_classifier.pkl"
    SCALER_PATH: str = "models/feature_scaler.pkl"
    MIN_AUDIO_DURATION: float = 1.0
    MAX_AUDIO_DURATION: float = 30.0
    
    # Audio Processing
    SAMPLE_RATE: int = 22050
    N_MFCC: int = 13
    
    class Config:
        env_file = ".env"
        case_sensitive = True
    
    @property
    def api_keys_list(self) -> List[str]:
        """Parse comma-separated API keys into a list"""
        return [key.strip() for key in self.API_KEYS.split(",")]


# Global settings instance
settings = Settings()
