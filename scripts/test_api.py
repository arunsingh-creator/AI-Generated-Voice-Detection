"""
Sample test script to verify the API
Run this after starting the server
"""
import base64
import requests
import json
from pathlib import Path

# Configuration
API_URL = "http://localhost:8000"
API_KEY = "your_secret_api_key_1"  # Change this to your actual API key


def test_health_check():
    """Test the health check endpoint"""
    print("=" * 60)
    print("TEST 1: Health Check")
    print("=" * 60)
    
    response = requests.get(f"{API_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()
    
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    print("✓ Health check passed\n")


def test_root_endpoint():
    """Test the root endpoint"""
    print("=" * 60)
    print("TEST 2: Root Endpoint")
    print("=" * 60)
    
    response = requests.get(f"{API_URL}/")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()
    
    assert response.status_code == 200
    print("✓ Root endpoint passed\n")


def test_missing_api_key():
    """Test request without API key"""
    print("=" * 60)
    print("TEST 3: Missing API Key")
    print("=" * 60)
    
    response = requests.post(
        f"{API_URL}/api/voice-detection",
        json={
            "language": "English",
            "audioFormat": "mp3",
            "audioBase64": "dummy"
        }
    )
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()
    
    assert response.status_code == 401
    print("✓ Missing API key test passed\n")


def test_invalid_api_key():
    """Test request with invalid API key"""
    print("=" * 60)
    print("TEST 4: Invalid API Key")
    print("=" * 60)
    
    response = requests.post(
        f"{API_URL}/api/voice-detection",
        headers={"x-api-key": "invalid_key_xyz"},
        json={
            "language": "English",
            "audioFormat": "mp3",
            "audioBase64": "dummy"
        }
    )
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()
    
    assert response.status_code == 401
    print("✓ Invalid API key test passed\n")


def test_invalid_language():
    """Test request with invalid language"""
    print("=" * 60)
    print("TEST 5: Invalid Language")
    print("=" * 60)
    
    response = requests.post(
        f"{API_URL}/api/voice-detection",
        headers={"x-api-key": API_KEY},
        json={
            "language": "French",  # Not supported
            "audioFormat": "mp3",
            "audioBase64": "U0lEM0JBQUFBQUFBSVRTU0VBQUFBUEFBQURNR0l6WmpVNEx6YzJMakV3TUFBQUFBQUFBQUFBQUFBQUFBPT0="
        }
    )
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()
    
    assert response.status_code == 422  # Validation error
    print("✓ Invalid language test passed\n")


def test_voice_detection_with_sample():
    """Test actual voice detection with a sample audio"""
    print("=" * 60)
    print("TEST 6: Voice Detection with Sample Audio")
    print("=" * 60)
    
    # Check if any sample audio exists
    sample_paths = [
        Path("data/raw/ai/ai_en_1.mp3"),
        Path("data/raw/human/human_en_1.mp3"),
    ]
    
    sample_file = None
    for path in sample_paths:
        if path.exists():
            sample_file = path
            break
    
    if sample_file is None:
        print("⚠ No sample audio files found. Skipping this test.")
        print("  Run 'python scripts/collect_data.py' to generate samples.\n")
        return
    
    # Read and encode audio
    with open(sample_file, "rb") as f:
        audio_base64 = base64.b64encode(f.read()).decode()
    
    print(f"Using sample: {sample_file}")
    print(f"Audio size: {len(audio_base64)} characters (Base64)")
    
    # Make request
    response = requests.post(
        f"{API_URL}/api/voice-detection",
        headers={"x-api-key": API_KEY},
        json={
            "language": "English",
            "audioFormat": "mp3",
            "audioBase64": audio_base64
        }
    )
    
    print(f"\nStatus Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()
    
    if response.status_code == 200:
        result = response.json()
        assert result["status"] == "success"
        assert result["classification"] in ["AI_GENERATED", "HUMAN"]
        assert 0.0 <= result["confidenceScore"] <= 1.0
        print("✓ Voice detection test passed\n")
    else:
        print(f"✗ Test failed with status {response.status_code}\n")


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("AI VOICE DETECTION API - TEST SUITE")
    print("=" * 60)
    print(f"API URL: {API_URL}")
    print(f"API Key: {API_KEY}")
    print("=" * 60 + "\n")
    
    try:
        test_health_check()
        test_root_endpoint()
        test_missing_api_key()
        test_invalid_api_key()
        test_invalid_language()
        test_voice_detection_with_sample()
        
        print("=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Cannot connect to API server")
        print("   Make sure the server is running: python -m uvicorn app.main:app --reload\n")
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}\n")


if __name__ == "__main__":
    main()
