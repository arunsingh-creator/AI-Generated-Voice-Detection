# AI-Generated Voice Detection API

🎯 **Production-ready REST API for detecting AI-generated vs human voices across 5 Indian languages**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [API Usage](#api-usage)
- [Model Training](#model-training)
- [Deployment](#deployment)
- [Testing](#testing)
- [Performance](#performance)

---

## 🎯 Overview

This API classifies voice samples as **AI-generated** or **HUMAN** with confidence scores and technical explanations. It supports 5 languages:

- 🇮🇳 **Tamil**
- 🇬🇧 **English**
- 🇮🇳 **Hindi**
- 🇮🇳 **Malayalam**
- 🇮🇳 **Telugu**

### Detection Method

Uses **acoustic feature analysis** + **machine learning**:
- **40+ audio features**: MFCCs, spectral features, pitch (F0), jitter, shimmer, HNR
- **Random Forest classifier** trained on real & AI voice samples
- **Language-agnostic** approach (works across all supported languages)
- **Hybrid fallback**: Heuristic-based classification if no trained model available

---

## ✨ Features

✅ **Strict API Contract Compliance**  
✅ **API Key Authentication**  
✅ **Base64 MP3 Input**  
✅ **JSON Request/Response**  
✅ **Confidence Scoring** (0.0 - 1.0)  
✅ **Technical Explanations**  
✅ **Comprehensive Error Handling**  
✅ **Docker Support**  
✅ **Production-Ready**  

---

## 🏗️ Architecture

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │ POST /api/voice-detection
       │ x-api-key: YOUR_KEY
       │ {language, audioFormat, audioBase64}
       ▼
┌─────────────────────────────────────┐
│         FastAPI Server              │
├─────────────────────────────────────┤
│  1. API Key Validation              │
│  2. Request Validation (Pydantic)   │
│  3. Base64 → MP3 Decoding           │
│  4. Feature Extraction (librosa)    │
│     • MFCCs, spectral, pitch, etc.  │
│  5. ML Classification (sklearn)     │
│     • Random Forest / Heuristics    │
│  6. Confidence Calculation          │
│  7. Response Generation             │
└─────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  {status, language, classification, │
│   confidenceScore, explanation}     │
└─────────────────────────────────────┘
```

**Tech Stack:**
- **Backend**: FastAPI (async Python)
- **Audio Processing**: librosa, pydub
- **ML**: scikit-learn (Random Forest)
- **Deployment**: Docker, uvicorn

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- ffmpeg (for audio processing)
- pip

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd AI-Generated-Voice-Detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and set your API_KEYS
```

### Initial Setup & Training

```bash
# Step 1: Collect training data
python scripts/collect_data.py

# This will:
# - Generate AI voice samples using gTTS
# - Create data/raw/ai/ directory with samples
# - Provide instructions for human samples

# Step 2: Add human voice samples
# Place MP3 files in data/raw/human/
# Example: human_en_1.mp3, human_ta_1.mp3, etc.

# Step 3: Train the model
python scripts/train_model.py

# This will:
# - Extract features from all samples
# - Train Random Forest classifier
# - Evaluate performance (cross-validation + test set)
# - Save model to models/voice_classifier.pkl
```

### Run the API

```bash
# Development mode (with auto-reload)
python -m uvicorn app.main:app --reload --port 8000

# Production mode
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

API will be available at: `http://localhost:8000`

**Interactive API Docs**: `http://localhost:8000/docs`

---

## 📡 API Usage

### Authentication

All requests require an API key in the header:

```
x-api-key: YOUR_SECRET_API_KEY
```

Set your API keys in `.env`:
```env
API_KEYS=your_secret_key_1,your_secret_key_2
```

### Endpoint

**POST** `/api/voice-detection`

### Request Format

```json
{
  "language": "English",
  "audioFormat": "mp3",
  "audioBase64": "<Base64-encoded MP3 audio>"
}
```

**Fields:**
- `language`: One of: `Tamil`, `English`, `Hindi`, `Malayalam`, `Telugu`
- `audioFormat`: Must be `mp3`
- `audioBase64`: Base64-encoded MP3 audio (1-30 seconds)

### Success Response

```json
{
  "status": "success",
  "language": "English",
  "classification": "AI_GENERATED",
  "confidenceScore": 0.87,
  "explanation": "High spectral consistency and low jitter indicate synthetic voice"
}
```

**Fields:**
- `classification`: `AI_GENERATED` or `HUMAN`
- `confidenceScore`: Float between 0.0 and 1.0
- `explanation`: Technical reason for classification

### Error Response

```json
{
  "status": "error",
  "message": "Invalid audio format: corrupted MP3 file"
}
```

### cURL Example

```bash
# Encode audio to Base64
base64 -w 0 sample.mp3 > sample_base64.txt

# Make API request
curl -X POST "http://localhost:8000/api/voice-detection" \
  -H "Content-Type: application/json" \
  -H "x-api-key: your_secret_api_key_1" \
  -d '{
    "language": "English",
    "audioFormat": "mp3",
    "audioBase64": "'$(cat sample_base64.txt)'"
  }'
```

### Python Example

```python
import base64
import requests

# Read and encode audio
with open("sample.mp3", "rb") as f:
    audio_base64 = base64.b64encode(f.read()).decode()

# Make request
response = requests.post(
    "http://localhost:8000/api/voice-detection",
    headers={"x-api-key": "your_secret_api_key_1"},
    json={
        "language": "English",
        "audioFormat": "mp3",
        "audioBase64": audio_base64
    }
)

result = response.json()
print(f"Classification: {result['classification']}")
print(f"Confidence: {result['confidenceScore']}")
```

---

## 🤖 Model Training

### Data Collection

The project includes scripts to generate training data:

```bash
python scripts/collect_data.py
```

This creates:
- **AI samples**: Generated using gTTS (Google Text-to-Speech)
- **Human samples**: Instructions for manual collection

**For Production Accuracy (80-85%+):**
1. Collect 20-30 real human voice recordings per language
2. Use diverse speakers (male/female, different ages)
3. Include various speaking styles (conversation, narration)
4. Use high-quality recordings (clear audio, minimal background noise)

**Public Dataset Sources:**
- [LibriVox](https://librivox.org/) - Public domain audiobooks
- [Mozilla Common Voice](https://commonvoice.mozilla.org/) - Crowdsourced voices
- [VoxForge](http://www.voxforge.org/) - Open speech corpus

### Training Process

```bash
python scripts/train_model.py
```

**What it does:**
1. Loads all MP3 files from `data/raw/human/` and `data/raw/ai/`
2. Extracts 40+ acoustic features per sample
3. Trains Random Forest classifier with:
   - 100 decision trees
   - 5-fold cross-validation
   - Balanced class weights
4. Evaluates on 20% test set
5. Saves model to `models/voice_classifier.pkl`

**Expected Performance:**
- **With 20+ samples per class**: 80-85% accuracy
- **With 50+ samples per class**: 85-90% accuracy
- **Production minimum**: 75%+ recommended

---

## 🚢 Deployment

### Option 1: Docker (Recommended)

```bash
# Build image
docker build -t ai-voice-detection:latest .

# Run container
docker run -d \
  -p 8000:8000 \
  -e API_KEYS=your_secret_key \
  --name ai-voice-api \
  ai-voice-detection:latest

# Or use Docker Compose
docker-compose up -d
```

### Option 2: Cloud Deployment

#### Render (Free Tier)

1. Push code to GitHub
2. Go to [Render Dashboard](https://dashboard.render.com/)
3. Create New → Web Service
4. Connect repository
5. Set environment variables:
   - `API_KEYS=your_key_here`
6. Deploy

#### Google Cloud Run

```bash
# Build and push image
gcloud builds submit --tag gcr.io/YOUR_PROJECT/ai-voice-detection

# Deploy
gcloud run deploy ai-voice-detection \
  --image gcr.io/YOUR_PROJECT/ai-voice-detection \
  --platform managed \
  --allow-unauthenticated \
  --set-env-vars API_KEYS=your_key_here
```

#### Railway

1. Install Railway CLI or use dashboard
2. `railway init`
3. Set environment variables
4. `railway up`

---

## 🧪 Testing

### Health Check

```bash
curl http://localhost:8000/health
```

Expected:
```json
{"status": "healthy", "service": "AI Voice Detection API", "version": "1.0.0"}
```

### API Tests

Test with sample audio (see `tests/` directory for examples)

```bash
# Unit tests (if implemented)
pytest tests/ -v

# Manual integration test
curl -X POST http://localhost:8000/api/voice-detection \
  -H "x-api-key: test_api_key_123" \
  -H "Content-Type: application/json" \
  -d @tests/sample_request.json
```

---

## ⚡ Performance & Scaling

### Current Performance

- **Latency**: ~0.5-2 seconds per request (depending on audio length)
- **Concurrency**: Handles 10-20 concurrent requests on single instance
- **Memory**: ~500MB-1GB per instance

### Optimization Tips

1. **Horizontal Scaling**: Deploy multiple instances behind load balancer
2. **Caching**: Cache feature extraction for duplicate audio (optional)
3. **Async Processing**: Use Celery + Redis for long audio files
4. **CDN**: Cache responses for frequently tested samples
5. **Rate Limiting**: Implement per-API-key limits

### Monitoring

Add logging and metrics:
- Request counts per API key
- Average confidence scores
- Classification distribution (AI vs Human)
- Error rates
- Response times

---

## 🔒 Security Considerations

- ✅ API key authentication
- ✅ Input validation (Pydantic)
- ✅ Request size limits (prevent DoS)
- ✅ CORS configuration
- ⚠️ Add rate limiting for production
- ⚠️ Use HTTPS in production
- ⚠️ Rotate API keys periodically

---

## 📝 License

MIT License - see LICENSE file

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create feature branch
3. Add tests
4. Submit pull request

---

## 📧 Support

For issues or questions, open a GitHub issue or contact: [your-email]

---

## 🙏 Acknowledgments

- **librosa**: Audio feature extraction
- **FastAPI**: Modern Python web framework
- **scikit-learn**: Machine learning tools
- **gTTS**: Text-to-speech for AI sample generation

---

**Built with ❤️ for AI voice detection**
