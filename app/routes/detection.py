"""
Voice detection API endpoint
"""
from fastapi import APIRouter, Depends, HTTPException, status
from app.models import (
    VoiceDetectionRequest,
    VoiceDetectionSuccessResponse,
    VoiceDetectionErrorResponse
)
from app.middleware import verify_api_key
from app.services import (
    decode_base64_to_waveform,
    get_classifier,
    AudioDecodingError,
    FeatureExtractionError,
    ClassificationError
)

# Use advanced feature extractor (63 features) to match enhanced model
try:
    from app.services.feature_extractor_advanced import extract_advanced_features, normalize_features
    USE_ADVANCED_FEATURES = True
except ImportError:
    from app.services.feature_extractor import extract_features
    USE_ADVANCED_FEATURES = False
from app.config import settings

router = APIRouter()


@router.post(
    "/api/voice-detection",
    response_model=VoiceDetectionSuccessResponse,
    responses={
        400: {"model": VoiceDetectionErrorResponse},
        401: {"model": VoiceDetectionErrorResponse},
        500: {"model": VoiceDetectionErrorResponse}
    },
    summary="Detect AI-Generated vs Human Voice",
    description="Analyzes an MP3 audio sample and classifies it as AI-generated or human voice"
)
async def detect_voice(
    request: VoiceDetectionRequest,
    api_key: str = Depends(verify_api_key)
) -> VoiceDetectionSuccessResponse:
    """
    Voice detection endpoint
    
    Accepts Base64-encoded MP3 audio and returns classification result.
    
    Args:
        request: Voice detection request with language, format, and audio data
        api_key: Validated API key from header
        
    Returns:
        Success response with classification and confidence
        
    Raises:
        HTTPException: For various error conditions
    """
    try:
        # Step 1: Decode Base64 audio to waveform
        waveform, duration = decode_base64_to_waveform(request.audioBase64)
        
        # Step 2: Extract acoustic features
        # Extract features using advanced extractor if available
        if USE_ADVANCED_FEATURES:
            features = extract_advanced_features(waveform, sr=settings.SAMPLE_RATE)
            features = normalize_features(features)
        else:
            features = extract_features(waveform, sr=settings.SAMPLE_RATE)
        
        # Step 3: Classify using ML model or heuristics
        classifier = get_classifier()
        classification, confidence, explanation = classifier.classify(features)
        
        # Step 4: Build response
        return VoiceDetectionSuccessResponse(
            status="success",
            language=request.language,
            classification=classification,
            confidenceScore=round(confidence, 2),
            explanation=explanation
        )
        
    except AudioDecodingError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    
    except FeatureExtractionError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Feature extraction failed: {str(e)}"
        )
    
    except ClassificationError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Classification failed: {str(e)}"
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )
