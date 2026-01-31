"""
Advanced Feature Extraction Module
Extracts enhanced acoustic features for improved AI vs Human voice classification
"""
import numpy as np
import librosa
from scipy import stats
from typing import Dict


def extract_advanced_features(waveform: np.ndarray, sr: int = 22050) -> Dict[str, float]:
    """
    Extract comprehensive acoustic features for AI voice detection
    
    Features include:
    - MFCCs and derivatives
    - Spectral features (centroid, bandwidth, rolloff, contrast)
    - Pitch features (F0, jitter, shimmer)
    - Voice quality (HNR, breathiness)
    - Temporal consistency metrics
    - Mel-spectrogram statistics
    
    Args:
        waveform: Audio waveform as numpy array
        sr: Sample rate
        
    Returns:
        Dictionary of feature names and values
    """
    features = {}
    
    # Ensure minimum length
    if len(waveform) < 2048:
        waveform = np.pad(waveform, (0, 2048 - len(waveform)))
    
    # ============================================================
    # 1. MFCC Features (13 coefficients + deltas)
    # ============================================================
    try:
        mfccs = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=13)
        
        # Statistics for each MFCC
        for i in range(13):
            features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
            features[f'mfcc_{i}_std'] = np.std(mfccs[i])
        
        # Delta MFCCs (velocity)
        mfcc_delta = librosa.feature.delta(mfccs)
        features['mfcc_delta_mean'] = np.mean(mfcc_delta)
        features['mfcc_delta_std'] = np.std(mfcc_delta)
    except Exception as e:
        print(f"Warning: MFCC extraction failed: {e}")
        for i in range(13):
            features[f'mfcc_{i}_mean'] = 0.0
            features[f'mfcc_{i}_std'] = 0.0
        features['mfcc_delta_mean'] = 0.0
        features['mfcc_delta_std'] = 0.0
    
    # ============================================================
    # 2. Spectral Features
    # ============================================================
    try:
        # Spectral centroid
        spectral_centroids = librosa.feature.spectral_centroid(y=waveform, sr=sr)[0]
        features['spectral_centroid_mean'] = np.mean(spectral_centroids)
        features['spectral_centroid_std'] = np.std(spectral_centroids)
        features['spectral_centroid_max'] = np.max(spectral_centroids)
        
        # Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=waveform, sr=sr)[0]
        features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
        features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=waveform, sr=sr)[0]
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        features['spectral_rolloff_std'] = np.std(spectral_rolloff)
        
        # Spectral contrast (7 bands)
        spectral_contrast = librosa.feature.spectral_contrast(y=waveform, sr=sr)
        for i in range(min(7, spectral_contrast.shape[0])):
            features[f'spectral_contrast_{i}'] = np.mean(spectral_contrast[i])
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(waveform)[0]
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)
    except Exception as e:
        print(f"Warning: Spectral feature extraction failed: {e}")
        # Set defaults
        features.update({
            'spectral_centroid_mean': 0.0,
            'spectral_centroid_std': 0.0,
            'spectral_centroid_max': 0.0,
            'spectral_bandwidth_mean': 0.0,
            'spectral_bandwidth_std': 0.0,
            'spectral_rolloff_mean': 0.0,
            'spectral_rolloff_std': 0.0,
            'zcr_mean': 0.0,
            'zcr_std': 0.0
        })
        for i in range(7):
            features[f'spectral_contrast_{i}'] = 0.0
    
    # ============================================================
    # 3. Pitch/F0 Features (CRITICAL for AI detection)
    # ============================================================
    try:
        # Extract pitch using piptrack
        pitches, magnitudes = librosa.piptrack(y=waveform, sr=sr)
        
        # Get pitch values where magnitude is significant
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:  # Valid pitch
                pitch_values.append(pitch)
        
        if len(pitch_values) > 0:
            pitch_values = np.array(pitch_values)
            features['pitch_mean'] = np.mean(pitch_values)
            features['pitch_std'] = np.std(pitch_values)
            features['pitch_min'] = np.min(pitch_values)
            features['pitch_max'] = np.max(pitch_values)
            features['pitch_range'] = features['pitch_max'] - features['pitch_min']
            features['pitch_median'] = np.median(pitch_values)
            
            # Jitter (pitch variation) - AI voices have very low jitter
            if len(pitch_values) > 1:
                pitch_diffs = np.abs(np.diff(pitch_values))
                features['jitter'] = np.mean(pitch_diffs) / (features['pitch_mean'] + 1e-8)
            else:
                features['jitter'] = 0.0
        else:
            # No pitch detected
            features.update({
                'pitch_mean': 0.0,
                'pitch_std': 0.0,
                'pitch_min': 0.0,
                'pitch_max': 0.0,
                'pitch_range': 0.0,
                'pitch_median': 0.0,
                'jitter': 0.0
            })
    except Exception as e:
        print(f"Warning: Pitch extraction failed: {e}")
        features.update({
            'pitch_mean': 0.0,
            'pitch_std': 0.0,
            'pitch_min': 0.0,
            'pitch_max': 0.0,
            'pitch_range': 0.0,
            'pitch_median': 0.0,
            'jitter': 0.0
        })
    
    # ============================================================
    # 4. Amplitude/Energy Features
    # ============================================================
    try:
        # RMS energy
        rms = librosa.feature.rms(y=waveform)[0]
        features['rms_mean'] = np.mean(rms)
        features['rms_std'] = np.std(rms)
        
        # Shimmer (amplitude variation) - AI voices have low shimmer
        if len(rms) > 1:
            rms_diffs = np.abs(np.diff(rms))
            features['shimmer'] = np.mean(rms_diffs) / (features['rms_mean'] + 1e-8)
        else:
            features['shimmer'] = 0.0
    except Exception as e:
        print(f"Warning: Energy feature extraction failed: {e}")
        features['rms_mean'] = 0.0
        features['rms_std'] = 0.0
        features['shimmer'] = 0.0
    
    # ============================================================
    # 5. Harmonic-to-Noise Ratio (HNR)
    # ============================================================
    try:
        # Separate harmonics and percussive components
        harmonic = librosa.effects.harmonic(waveform)
        percussive = librosa.effects.percussive(waveform)
        
        harmonic_power = np.sum(harmonic ** 2)
        noise_power = np.sum(percussive ** 2)
        
        if noise_power > 0:
            features['hnr'] = 10 * np.log10(harmonic_power / (noise_power + 1e-8))
        else:
            features['hnr'] = 100.0  # Very high HNR
    except Exception as e:
        print(f"Warning: HNR extraction failed: {e}")
        features['hnr'] = 0.0
    
    # ============================================================
    # 6. Chroma Features
    # ============================================================
    try:
        chroma = librosa.feature.chroma_stft(y=waveform, sr=sr)
        features['chroma_mean'] = np.mean(chroma)
        features['chroma_std'] = np.std(chroma)
    except Exception as e:
        print(f"Warning: Chroma extraction failed: {e}")
        features['chroma_mean'] = 0.0
        features['chroma_std'] = 0.0
    
    # ============================================================
    # 7. Mel-Spectrogram Statistics
    # ============================================================
    try:
        mel_spec = librosa.feature.melspectrogram(y=waveform, sr=sr, n_mels=40)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        features['mel_spec_mean'] = np.mean(mel_spec_db)
        features['mel_spec_std'] = np.std(mel_spec_db)
        features['mel_spec_skewness'] = stats.skew(mel_spec_db.flatten())
        features['mel_spec_kurtosis'] = stats.kurtosis(mel_spec_db.flatten())
    except Exception as e:
        print(f"Warning: Mel-spectrogram extraction failed: {e}")
        features['mel_spec_mean'] = 0.0
        features['mel_spec_std'] = 0.0
        features['mel_spec_skewness'] = 0.0
        features['mel_spec_kurtosis'] = 0.0
    
    # ============================================================
    # 8. Temporal Consistency (AI voices are too consistent)
    # ============================================================
    try:
        # Split into frames and measure consistency
        frame_length = sr // 10  # 100ms frames
        num_frames = len(waveform) // frame_length
        
        if num_frames > 2:
            frame_energies = []
            frame_zcrs = []
            
            for i in range(num_frames):
                frame = waveform[i * frame_length:(i + 1) * frame_length]
                if len(frame) > 0:
                    frame_energies.append(np.mean(frame ** 2))
                    frame_zcrs.append(np.mean(librosa.zero_crossings(frame)))
            
            if len(frame_energies) > 0:
                features['temporal_energy_variation'] = np.std(frame_energies)
                features['temporal_zcr_variation'] = np.std(frame_zcrs)
            else:
                features['temporal_energy_variation'] = 0.0
                features['temporal_zcr_variation'] = 0.0
        else:
            features['temporal_energy_variation'] = 0.0
            features['temporal_zcr_variation'] = 0.0
    except Exception as e:
        print(f"Warning: Temporal feature extraction failed: {e}")
        features['temporal_energy_variation'] = 0.0
        features['temporal_zcr_variation'] = 0.0
    
    return features


def normalize_features(features: Dict[str, float]) -> Dict[str, float]:
    """Normalize feature values to reasonable ranges"""
    normalized = {}
    
    for key, value in features.items():
        # Replace inf and nan with 0
        if np.isinf(value) or np.isnan(value):
            normalized[key] = 0.0
        else:
            normalized[key] = float(value)
    
    return normalized
