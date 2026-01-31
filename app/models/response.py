"""
Pydantic models for API responses
"""
from pydantic import BaseModel, Field
from typing import Literal


class VoiceDetectionSuccessResponse(BaseModel):
    """Success response model for voice detection"""
    
    status: Literal["success"] = "success"
    language: str = Field(..., description="Language from the request")
    classification: Literal["AI_GENERATED", "HUMAN"] = Field(
        ...,
        description="Classification result"
    )
    confidenceScore: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0"
    )
    explanation: str = Field(
        ...,
        description="Brief technical explanation of the classification"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "success",
                    "language": "English",
                    "classification": "AI_GENERATED",
                    "confidenceScore": 0.87,
                    "explanation": "High spectral consistency and low jitter indicate synthetic voice"
                }
            ]
        }
    }


class VoiceDetectionErrorResponse(BaseModel):
    """Error response model"""
    
    status: Literal["error"] = "error"
    message: str = Field(..., description="Error message describing what went wrong")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "error",
                    "message": "Invalid audio format: corrupted MP3 file"
                }
            ]
        }
    }
