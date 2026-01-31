"""
Sample usage script demonstrating how to use the API
"""
import base64
import requests
import json
from pathlib import Path


def classify_audio_file(audio_path: str, language: str, api_key: str, api_url: str = "http://localhost:8000"):
    """
    Classify an audio file as AI-generated or human
    
    Args:
        audio_path: Path to MP3 audio file
        language: Language of the audio (Tamil, English, Hindi, Malayalam, Telugu)
        api_key: Your API key
        api_url: API base URL (default: http://localhost:8000)
    
    Returns:
        Dictionary with classification result
    """
    # Read and encode audio
    with open(audio_path, "rb") as f:
        audio_base64 = base64.b64encode(f.read()).decode()
    
    # Make API request
    response = requests.post(
        f"{api_url}/api/voice-detection",
        headers={
            "x-api-key": api_key,
            "Content-Type": "application/json"
        },
        json={
            "language": language,
            "audioFormat": "mp3",
            "audioBase64": audio_base64
        }
    )
    
    # Handle response
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"API Error: {response.json()}")


def main():
    """Example usage"""
    # Configuration
    API_KEY = "test_api_key_123"  # Replace with your actual API key
    API_URL = "http://localhost:8000"
    
    # Example 1: Classify a single file
    print("=" * 60)
    print("Example 1: Classify Single Audio File")
    print("=" * 60)
    
    audio_file = "data/raw/ai/ai_en_1.mp3"
    
    if Path(audio_file).exists():
        try:
            result = classify_audio_file(
                audio_path=audio_file,
                language="English",
                api_key=API_KEY,
                api_url=API_URL
            )
            
            print(f"\nFile: {audio_file}")
            print(f"Language: {result['language']}")
            print(f"Classification: {result['classification']}")
            print(f"Confidence: {result['confidenceScore']:.2%}")
            print(f"Explanation: {result['explanation']}\n")
            
        except Exception as e:
            print(f"Error: {e}\n")
    else:
        print(f"Sample file not found: {audio_file}")
        print("Run 'python scripts/collect_data.py' to generate samples.\n")
    
    # Example 2: Batch classification
    print("=" * 60)
    print("Example 2: Batch Classification")
    print("=" * 60)
    
    sample_files = [
        {"path": "data/raw/ai/ai_en_1.mp3", "language": "English"},
        {"path": "data/raw/ai/ai_ta_1.mp3", "language": "Tamil"},
        {"path": "data/raw/human/human_en_1.mp3", "language": "English"},
    ]
    
    results = []
    for sample in sample_files:
        if Path(sample["path"]).exists():
            try:
                result = classify_audio_file(
                    audio_path=sample["path"],
                    language=sample["language"],
                    api_key=API_KEY,
                    api_url=API_URL
                )
                results.append({
                    "file": sample["path"],
                    "classification": result["classification"],
                    "confidence": result["confidenceScore"]
                })
            except Exception as e:
                print(f"Error processing {sample['path']}: {e}")
    
    if results:
        print("\nBatch Results:")
        print("-" * 60)
        for r in results:
            print(f"{Path(r['file']).name:30} | {r['classification']:13} | {r['confidence']:.2%}")
        print()
    else:
        print("No sample files found for batch processing.\n")
    
    # Example 3: Custom audio recording
    print("=" * 60)
    print("Example 3: Classify Your Own Recording")
    print("=" * 60)
    print("""
To classify your own voice recording:

1. Record audio (5-10 seconds) and save as MP3
2. Use the classify_audio_file() function:

    result = classify_audio_file(
        audio_path="my_recording.mp3",
        language="English",  # Or Tamil, Hindi, Malayalam, Telugu
        api_key="YOUR_API_KEY"
    )
    
    print(f"Classification: {result['classification']}")
    print(f"Confidence: {result['confidenceScore']:.2%}")
    """)


if __name__ == "__main__":
    main()
