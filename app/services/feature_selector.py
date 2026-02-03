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

# Decision logic: Prefer Advanced features (63) since we deployed the advanced model
# Check if advanced extractor is importable (it should be)
try:
    from app.services.feature_extractor_advanced import extract_advanced_features, normalize_features
    extract_features = lambda waveform, sr: normalize_features(extract_advanced_features(waveform, sr))
    USE_BASIC = False
    print("🔧 Feature Selector: Using ADVANCED features (63) - Matches deployed model")
except ImportError:
    from app.services.feature_extractor import extract_features
    USE_BASIC = True
    print("🔧 Feature Selector: Fallback to BASIC features (48) - Advanced module missing")

USE_ADVANCED = not USE_BASIC

