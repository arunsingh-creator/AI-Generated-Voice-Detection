"""
Pydantic models for API requests
"""
from pydantic import BaseModel, Field, field_validator
from typing import Literal


class VoiceDetectionRequest(BaseModel):
    """Request model for voice detection endpoint"""
    
    language: Literal["Tamil", "English", "Hindi", "Malayalam", "Telugu"] = Field(
        ...,
        description="Language of the audio sample"
    )
    
    audioFormat: Literal["mp3"] = Field(
        ...,
        description="Audio format - only MP3 is supported"
    )
    
    audioBase64: str = Field(
        ...,
        description="Base64-encoded MP3 audio data",
        min_length=100
    )
    
    @field_validator("audioBase64")
    @classmethod
    def validate_base64(cls, v: str) -> str:
        """Validate that the string looks like valid base64"""
        import re
        if not re.match(r'^[A-Za-z0-9+/]+={0,2}$', v.replace('\n', '').replace('\r', '')):
            raise ValueError("Invalid Base64 encoding")
        return v
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "language": "English",
                    "audioFormat": "mp3",
                    "audioBase64": "SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU4Ljc2LjEwMAAAAAAAAAAAAAAA..."
                }
            ]
        }
    }
