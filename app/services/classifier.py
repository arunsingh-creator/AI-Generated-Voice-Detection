"""
Voice classifier service - classifies audio as AI-generated or human
"""
import os
import numpy as np
import joblib
from typing import Tuple, Dict
from app.config import settings


class ClassificationError(Exception):
    """Custom exception for classification errors"""
    pass


class VoiceClassifier:
    """Voice classifier using trained ML model or heuristic fallback"""
    
    def __init__(self):
        """Initialize classifier and load model if available"""
        self.model = None
        self.scaler = None
        self.use_heuristic = True
        
        # Try to load trained model
        if os.path.exists(settings.MODEL_PATH):
            try:
                self.model = joblib.load(settings.MODEL_PATH)
                self.use_heuristic = False
                print(f"✓ Loaded trained model from {settings.MODEL_PATH}")
            except Exception as e:
                print(f"⚠ Failed to load model: {e}. Using heuristic fallback.")
        
        # Try to load scaler
        if os.path.exists(settings.SCALER_PATH):
            try:
                self.scaler = joblib.load(settings.SCALER_PATH)
                print(f"✓ Loaded feature scaler from {settings.SCALER_PATH}")
            except Exception as e:
                print(f"⚠ Failed to load scaler: {e}")
    
    def classify(self, features: Dict[str, float]) -> Tuple[str, float, str]:
        """
        Classify voice as AI_GENERATED or HUMAN
        
        Args:
            features: Dictionary of extracted acoustic features
            
        Returns:
            Tuple of (classification, confidence_score, explanation)
            
        Raises:
            ClassificationError: If classification fails
        """
        try:
            if self.use_heuristic:
                return self._classify_heuristic(features)
            else:
                return self._classify_ml(features)
        except Exception as e:
            raise ClassificationError(f"Classification failed: {str(e)}")
    
    def _classify_ml(self, features: Dict[str, float]) -> Tuple[str, float, str]:
        """Classify using trained ML model"""
        # Convert features to ordered array
        feature_names = sorted(features.keys())
        feature_vector = np.array([features[name] for name in feature_names]).reshape(1, -1)
        
        # Scale features if scaler is available
        if self.scaler is not None:
            feature_vector = self.scaler.transform(feature_vector)
        
        # Get prediction and probability
        prediction = self.model.predict(feature_vector)[0]
        probabilities = self.model.predict_proba(feature_vector)[0]
        
        # Map prediction to classification
        if prediction == 1:  # AI-generated
            classification = "AI_GENERATED"
            confidence = float(probabilities[1])
        else:  # Human
            classification = "HUMAN"
            confidence = float(probabilities[0])
        
        # Generate explanation based on key features
        explanation = self._generate_ml_explanation(features, classification)
        
        return classification, confidence, explanation
    
    def _classify_heuristic(self, features: Dict[str, float]) -> Tuple[str, float, str]:
        """
        Classify using feature-based heuristics
        
        AI-generated voices typically have:
        - Lower jitter (more stable pitch)
        - Lower shimmer (more stable amplitude)
        - Higher harmonic-to-noise ratio
        - More consistent spectral features
        - Less natural variation in prosody
        """
        # Extract key discriminative features
        jitter = features.get('jitter', 0.0)
        shimmer = features.get('shimmer', 0.0)
        hnr = features.get('hnr', 0.0)
        pitch_std = features.get('pitch_std', 0.0)
        spectral_centroid_std = features.get('spectral_centroid_std', 0.0)
        
        # Initialize score (0 = human, 1 = AI)
        ai_score = 0.0
        reasons = []
        
        # Rule 1: Very low jitter suggests AI (human voices have natural pitch variation)
        if jitter < 0.005:  # Threshold based on typical human jitter ~0.01-0.03
            ai_score += 0.25
            reasons.append("extremely stable pitch")
        elif jitter > 0.02:
            ai_score -= 0.2
            reasons.append("natural pitch variation")
        
        # Rule 2: Very low shimmer suggests AI
        if shimmer < 0.01:  # Threshold based on typical human shimmer ~0.02-0.05
            ai_score += 0.25
            reasons.append("uniform amplitude")
        elif shimmer > 0.03:
            ai_score -= 0.2
            reasons.append("natural amplitude variation")
        
        # Rule 3: Unusually high HNR suggests AI (very clean signal)
        if hnr > 20:  # Very high harmonic content
            ai_score += 0.2
            reasons.append("high harmonic purity")
        elif hnr < 10:
            ai_score -= 0.15
            reasons.append("natural breathiness")
        
        # Rule 4: Low pitch variability suggests AI
        if pitch_std < 10:  # Hz
            ai_score += 0.15
            reasons.append("monotone pitch pattern")
        elif pitch_std > 30:
            ai_score -= 0.15
            reasons.append("expressive pitch range")
        
        # Rule 5: Very consistent spectral features suggest AI
        if spectral_centroid_std < 200:
            ai_score += 0.15
            reasons.append("consistent spectral profile")
        
        # Normalize score to [0, 1]
        ai_score = max(0.0, min(1.0, ai_score + 0.5))
        
        # Classify based on threshold
        if ai_score > 0.5:
            classification = "AI_GENERATED"
            confidence = ai_score
            explanation = f"Synthetic indicators: {', '.join(reasons[:3])}"
        else:
            classification = "HUMAN"
            confidence = 1.0 - ai_score
            explanation = f"Natural voice characteristics: {', '.join(reasons[:3])}"
        
        return classification, confidence, explanation
    
    def _generate_ml_explanation(self, features: Dict[str, float], classification: str) -> str:
        """Generate explanation based on feature importances"""
        # Get top 3 most discriminative features (simplified version)
        key_features = {
            'jitter': features.get('jitter', 0.0),
            'shimmer': features.get('shimmer', 0.0),
            'hnr': features.get('hnr', 0.0),
            'pitch_std': features.get('pitch_std', 0.0)
        }
        
        if classification == "AI_GENERATED":
            if key_features['jitter'] < 0.01 and key_features['shimmer'] < 0.02:
                return "Low prosodic variability and high signal consistency indicate synthetic voice"
            else:
                return "Spectral and temporal patterns match AI-generated voice characteristics"
        else:
            if key_features['jitter'] > 0.02 and key_features['shimmer'] > 0.03:
                return "Natural prosodic variation and spectral irregularities indicate human voice"
            else:
                return "Acoustic features align with natural human speech patterns"


# Global classifier instance
_classifier = None


def get_classifier() -> VoiceClassifier:
    """Get or create global classifier instance"""
    global _classifier
    if _classifier is None:
        _classifier = VoiceClassifier()
    return _classifier
