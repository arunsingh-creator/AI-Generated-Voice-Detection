"""
Feature extraction service - extracts acoustic features from audio
"""
import numpy as np
import librosa
from typing import Dict
from app.config import settings


class FeatureExtractionError(Exception):
    """Custom exception for feature extraction errors"""
    pass


def extract_features(waveform: np.ndarray, sr: int = None) -> Dict[str, float]:
    """
    Extract comprehensive acoustic features from audio waveform
    
    Features extracted:
    - MFCCs (Mel-Frequency Cepstral Coefficients)
    - Spectral features (centroid, bandwidth, rolloff)
    - Pitch features (F0 mean, std, range)
    - Temporal features (jitter, shimmer, zero-crossing rate)
    - Energy features
    
    Args:
        waveform: Audio waveform as numpy array
        sr: Sample rate (defaults to settings.SAMPLE_RATE)
        
    Returns:
        Dictionary of extracted features
        
    Raises:
        FeatureExtractionError: If extraction fails
    """
    if sr is None:
        sr = settings.SAMPLE_RATE
    
    try:
        features = {}
        
        # ===== MFCC Features =====
        mfccs = librosa.feature.mfcc(
            y=waveform,
            sr=sr,
            n_mfcc=settings.N_MFCC
        )
        
        # Statistical aggregations of MFCCs
        for i in range(settings.N_MFCC):
            features[f'mfcc_{i}_mean'] = float(np.mean(mfccs[i]))
            features[f'mfcc_{i}_std'] = float(np.std(mfccs[i]))
        
        # ===== Spectral Features =====
        spectral_centroids = librosa.feature.spectral_centroid(y=waveform, sr=sr)[0]
        features['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
        features['spectral_centroid_std'] = float(np.std(spectral_centroids))
        
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=waveform, sr=sr)[0]
        features['spectral_bandwidth_mean'] = float(np.mean(spectral_bandwidth))
        features['spectral_bandwidth_std'] = float(np.std(spectral_bandwidth))
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=waveform, sr=sr)[0]
        features['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))
        features['spectral_rolloff_std'] = float(np.std(spectral_rolloff))
        
        # ===== Zero-Crossing Rate =====
        zcr = librosa.feature.zero_crossing_rate(waveform)[0]
        features['zcr_mean'] = float(np.mean(zcr))
        features['zcr_std'] = float(np.std(zcr))
        
        # ===== Pitch (F0) Features =====
        # Extract pitch using probabilistic YIN algorithm
        f0 = librosa.yin(
            waveform,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr
        )
        
        # Filter out unvoiced frames (pitch = NaN or 0)
        voiced_f0 = f0[~np.isnan(f0)]
        voiced_f0 = voiced_f0[voiced_f0 > 0]
        
        if len(voiced_f0) > 0:
            features['pitch_mean'] = float(np.mean(voiced_f0))
            features['pitch_std'] = float(np.std(voiced_f0))
            features['pitch_min'] = float(np.min(voiced_f0))
            features['pitch_max'] = float(np.max(voiced_f0))
            features['pitch_range'] = features['pitch_max'] - features['pitch_min']
            
            # Jitter (pitch variability) - normalized
            if len(voiced_f0) > 1:
                pitch_diffs = np.abs(np.diff(voiced_f0))
                features['jitter'] = float(np.mean(pitch_diffs) / features['pitch_mean'])
            else:
                features['jitter'] = 0.0
        else:
            # No voiced frames detected
            features['pitch_mean'] = 0.0
            features['pitch_std'] = 0.0
            features['pitch_min'] = 0.0
            features['pitch_max'] = 0.0
            features['pitch_range'] = 0.0
            features['jitter'] = 0.0
        
        # ===== Energy Features =====
        rms_energy = librosa.feature.rms(y=waveform)[0]
        features['rms_energy_mean'] = float(np.mean(rms_energy))
        features['rms_energy_std'] = float(np.std(rms_energy))
        
        # Shimmer (amplitude variability) - normalized
        if len(rms_energy) > 1 and features['rms_energy_mean'] > 0:
            energy_diffs = np.abs(np.diff(rms_energy))
            features['shimmer'] = float(np.mean(energy_diffs) / features['rms_energy_mean'])
        else:
            features['shimmer'] = 0.0
        
        # ===== Harmonic-to-Noise Ratio (HNR) =====
        # Separate harmonic and percussive components
        harmonic, percussive = librosa.effects.hpss(waveform)
        harmonic_energy = np.sum(harmonic ** 2)
        noise_energy = np.sum(percussive ** 2)
        
        if noise_energy > 0:
            features['hnr'] = float(10 * np.log10(harmonic_energy / noise_energy))
        else:
            features['hnr'] = 100.0  # Very high HNR if no noise
        
        # ===== Spectral Contrast =====
        spectral_contrast = librosa.feature.spectral_contrast(y=waveform, sr=sr)
        features['spectral_contrast_mean'] = float(np.mean(spectral_contrast))
        features['spectral_contrast_std'] = float(np.std(spectral_contrast))
        
        # ===== Chroma Features =====
        chroma = librosa.feature.chroma_stft(y=waveform, sr=sr)
        features['chroma_mean'] = float(np.mean(chroma))
        features['chroma_std'] = float(np.std(chroma))
        
        return features
        
    except Exception as e:
        raise FeatureExtractionError(f"Failed to extract features: {str(e)}")


def get_feature_vector(features: Dict[str, float]) -> np.ndarray:
    """
    Convert feature dictionary to ordered numpy array
    
    Args:
        features: Dictionary of features
        
    Returns:
        Numpy array of features in consistent order
    """
    # Define expected feature order (for model consistency)
    feature_names = sorted(features.keys())
    return np.array([features[name] for name in feature_names])
