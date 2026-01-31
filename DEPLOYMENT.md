# 🚀 Deployment with Auto-Model Training

## Overview

The application is configured to **automatically train the model on deployment**. This means:
- ✅ No need to commit large model files to GitHub
- ✅ Fresh model trained on every deployment
- ✅ Uses latest training scripts and data collection

---

## How It Works

### Deployment Flow

1. **Push Code to GitHub**
   ```bash
   git push origin main
   ```

2. **Render Pulls Latest Code**
   - Downloads repository
   - Installs dependencies from `requirements.txt`

3. **Dockerfile Builds Container**
   - Copies `app/` and `scripts/` directories
   - Creates `models/` and `data/raw/` directories

4. **Auto-Training Script Runs** (`scripts/deploy_train.py`)
   - Checks if models exist
   - If not, generates AI voice samples using gTTS
   - Trains model using `scripts/train_model.py`
   - Saves `models/voice_classifier.pkl` and `models/feature_scaler.pkl`

5. **API Server Starts**
   - Loads trained model
   - Ready to serve requests

---

## Deployment Command

**In Dockerfile:**
```dockerfile
CMD python scripts/deploy_train.py && uvicorn app.main:app --host 0.0.0.0 --port 8000
```

This command:
1. Runs `deploy_train.py` (trains model if needed)
2. Then starts the API with `uvicorn`

---

## Expected Deployment Timeline

| Stage | Duration | What Happens |
|-------|----------|--------------|
| Build | 2-3 min | Install dependencies |
| Training | 1-2 min | Generate samples, train model |
| Startup | 10-30 sec | Load model, start API |
| **Total** | **3-6 min** | Ready to serve |

---

## Training Data on Deployment

### Current Configuration (Default)

**Generated Samples:**
- AI samples: gTTS (Google Text-to-Speech)
- Human samples: gTTS with variation (placeholder)
- Languages: English, Tamil, Hindi, Malayalam, Telugu
- Expected accuracy: **40-50%** (both classes are AI-generated)

### To Improve Accuracy on Deployment

**Option 1: Include Training Data in Repository**
```bash
# 1. Commit sample audio files
git add data/raw/ai/*.mp3
git add data/raw/human/*.mp3

# 2. Update .gitignore (remove data exclusion)
# Comment out or remove: data/raw/

# 3. Push to GitHub
git push origin main
```

**Option 2: Download Datasets on Deployment**
Update `scripts/deploy_train.py` to:
- Download Mozilla Common Voice samples
- Fetch from external API/S3
- Use environment variable URLs

---

## Monitoring Deployment

### Check Deployment Logs (Render)

1. Go to Render Dashboard
2. Select your service
3. Click "Logs" tab
4. Look for:

```
========================================================
DEPLOYMENT MODEL TRAINING
========================================================
⚠ Models not found - running training...
📢 No training data found - generating samples...
📢 Training model...
✓ Model training complete!
========================================================
```

### Verify Model Loaded

**Check API Health:**
```bash
curl https://your-app.onrender.com/health
```

**Check Logs for:**
```
✓ Loaded trained model from models/voice_classifier.pkl
✓ Loaded feature scaler from models/feature_scaler.pkl
```

---

## Troubleshooting

### Issue: Training Takes Too Long (>5 min)

**Causes:**
- Large dataset
- Hyperparameter tuning enabled

**Solution:**
```python
# In scripts/deploy_train.py, use basic training instead:
from scripts.train_model import main as train_basic
train_basic()  # Faster, no GridSearchCV
```

### Issue: Out of Memory During Training

**Solution:**
Reduce dataset size in `scripts/collect_data.py`:
```python
# Generate fewer samples
SAMPLE_TEXTS = {
    "en": ["Sample 1", "Sample 2"],  # Reduce from 7 to 2
    # ...
}
```

### Issue: Model Not Loading

**Check Logs for:**
```
✗ Failed to load model: [error message]
⚠ Using heuristic fallback
```

**This is OK** - API will use heuristic classification (~70% accuracy)

---

## Alternative: Pre-Trained Models

If auto-training is problematic, commit pre-trained models:

### Step 1: Train Locally
```bash
python scripts/train_model_enhanced.py
```

### Step 2: Commit Models
```bash
# Edit .gitignore - comment out:
# models/*.pkl
# models/*.joblib

git add models/voice_classifier.pkl
git add models/feature_scaler.pkl
git commit -m "chore: Add pre-trained models"
git push origin main
```

### Step 3: Update Dockerfile
```dockerfile
# Remove auto-training
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Environment Variables

**For deployment, set these in Render:**

```env
# API Configuration
API_KEYS=your_secret_api_key_1,your_secret_api_key_2
PORT=8000
ENVIRONMENT=production

# Model Paths (default paths work)
MODEL_PATH=models/voice_classifier.pkl
SCALER_PATH=models/feature_scaler.pkl

# Training Configuration (optional)
SKIP_TRAINING=false  # Set to 'true' to skip auto-training
```

---

## Benefits of Auto-Training

✅ **No Large Files in Git**
- Model files can be 10-100 MB
- Git stays lightweight
- Faster clone/pull operations

✅ **Always Fresh Model**
- Uses latest training code
- Incorporates any script improvements
- No version mismatch

✅ **Reproducible**
- Same code + same data = same model
- No "works on my machine" issues

---

## Current Deployment Status

**GitHub**: Models gitignored (not committed)  
**Render**: Auto-trains on deployment  
**Training Script**: `scripts/deploy_train.py`  
**Expected Accuracy**: 40-50% (placeholder data)  
**To Improve**: Add real human voice samples

---

**Ready to Deploy!** 🚀

Just push your code and Render will handle the rest.
