"""
Environment-aware feature extraction
Uses basic features in production (deployment), advanced features in development
"""
import os

# Detect environment
ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')
IS_PRODUCTION = ENVIRONMENT == 'production'

if IS_PRODUCTION:
    # Production: Use basic features (48) to match deployed model
    from app.services.feature_extractor import extract_features
    USE_ADVANCED = False
    print("🔧 Production mode: Using basic feature extractor (48 features)")
else:
    # Development: Use advanced features (63)
    try:
        from app.services.feature_extractor_advanced import extract_advanced_features, normalize_features
        extract_features = lambda waveform, sr: normalize_features(extract_advanced_features(waveform, sr))
        USE_ADVANCED = True
        print("🔧 Development mode: Using advanced feature extractor (63 features)")
    except ImportError:
        from app.services.feature_extractor import extract_features
        USE_ADVANCED = False
        print("🔧 Fallback: Using basic feature extractor (48 features)")
