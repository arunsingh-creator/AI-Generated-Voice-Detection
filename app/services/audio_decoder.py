"""
Audio decoding service - converts Base64 MP3 to numpy waveform
"""
import base64
import io
import numpy as np
from pydub import AudioSegment
import librosa
from app.config import settings


class AudioDecodingError(Exception):
    """Custom exception for audio decoding errors"""
    pass


def decode_base64_to_waveform(audio_base64: str) -> tuple[np.ndarray, float]:
    """
    Decode Base64-encoded MP3 audio to numpy waveform
    
    Args:
        audio_base64: Base64-encoded MP3 audio string
        
    Returns:
        Tuple of (waveform as numpy array, duration in seconds)
        
    Raises:
        AudioDecodingError: If decoding fails or audio is invalid
    """
    try:
        # Decode Base64 to bytes
        audio_bytes = base64.b64decode(audio_base64)
        
        # Load MP3 using pydub
        audio_segment = AudioSegment.from_file(
            io.BytesIO(audio_bytes),
            format="mp3"
        )
        
        # Get duration
        duration = len(audio_segment) / 1000.0  # Convert ms to seconds
        
        # Validate duration
        if duration < settings.MIN_AUDIO_DURATION:
            raise AudioDecodingError(
                f"Audio too short: {duration:.2f}s (minimum: {settings.MIN_AUDIO_DURATION}s)"
            )
        
        if duration > settings.MAX_AUDIO_DURATION:
            raise AudioDecodingError(
                f"Audio too long: {duration:.2f}s (maximum: {settings.MAX_AUDIO_DURATION}s)"
            )
        
        # Convert to numpy array
        samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
        
        # Normalize to [-1, 1]
        if audio_segment.sample_width == 2:  # 16-bit
            samples = samples / 32768.0
        elif audio_segment.sample_width == 4:  # 32-bit
            samples = samples / 2147483648.0
        
        # Convert to mono if stereo
        if audio_segment.channels == 2:
            samples = samples.reshape((-1, 2)).mean(axis=1)
        
        # Resample to target sample rate using librosa
        original_sr = audio_segment.frame_rate
        if original_sr != settings.SAMPLE_RATE:
            samples = librosa.resample(
                samples,
                orig_sr=original_sr,
                target_sr=settings.SAMPLE_RATE
            )
        
        return samples, duration
        
    except base64.binascii.Error as e:
        raise AudioDecodingError(f"Invalid Base64 encoding: {str(e)}")
    except Exception as e:
        raise AudioDecodingError(f"Failed to decode audio: {str(e)}")
