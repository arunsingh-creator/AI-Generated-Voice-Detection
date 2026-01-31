"""
Environment-aware feature extraction
Defaults to BASIC features (48) for production safety
"""
import os
from pathlib import Path

# Multiple ways to detect production environment
RENDER_SERVICE = os.getenv('RENDER_SERVICE_NAME')  # Render sets this
RENDER_INSTANCE = os.getenv('RENDER_INSTANCE_ID')   # Render sets this
ENV_VAR = os.getenv('ENVIRONMENT', '').lower()
PORT = os.getenv('PORT', '8000')

# Detect if running on Render or production
IS_RENDER = RENDER_SERVICE is not None or RENDER_INSTANCE is not None
IS_PRODUCTION = ENV_VAR == 'production' or IS_RENDER or PORT != '8000'

# Check if models directory has a trained model (indicator of local dev with enhanced model)
models_dir = Path("models")
has_enhanced_model = (models_dir / "voice_classifier.pkl").exists()

# Decision logic: Use basic features by default (safer)
# Only use advanced if explicitly in development AND have enhanced model
USE_BASIC = True  # Default to basic for safety

if not IS_PRODUCTION and not IS_RENDER:
    # Likely local development - check if we have enhanced model
    try:
        from app.services.feature_extractor_advanced import extract_advanced_features, normalize_features
        extract_features = lambda waveform, sr: normalize_features(extract_advanced_features(waveform, sr))
        USE_BASIC = False
        print("🔧 Development: Using advanced features (63)")
    except ImportError:
        from app.services.feature_extractor import extract_features
        print("🔧 Development fallback: Using basic features (48)")
else:
    # Production/Render - always use basic features
    from app.services.feature_extractor import extract_features
    print(f"🔧 Production (Render={IS_RENDER}): Using basic features (48)")

USE_ADVANCED = not USE_BASIC

