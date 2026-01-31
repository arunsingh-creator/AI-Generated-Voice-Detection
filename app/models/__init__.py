"""
Models package initialization
"""
from app.models.request import VoiceDetectionRequest
from app.models.response import VoiceDetectionSuccessResponse, VoiceDetectionErrorResponse

__all__ = [
    "VoiceDetectionRequest",
    "VoiceDetectionSuccessResponse",
    "VoiceDetectionErrorResponse",
]
