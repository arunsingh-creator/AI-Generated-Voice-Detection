"""
Comprehensive System Verification Script
Tests all components of the enhanced AI Voice Detection system
"""
import os
import sys
from pathlib import Path
import numpy as np

# Colors for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_status(message, status='info'):
    """Print colored status messages"""
    if status == 'success':
        print(f"{GREEN}✓{RESET} {message}")
    elif status == 'error':
        print(f"{RED}✗{RESET} {message}")
    elif status == 'warning':
        print(f"{YELLOW}⚠{RESET} {message}")
    else:
        print(f"{BLUE}ℹ{RESET} {message}")


def check_dependencies():
    """Check if all required dependencies are installed"""
    print("\n" + "="*60)
    print("1. CHECKING DEPENDENCIES")
    print("="*60)
    
    required = {
        'numpy': 'Core numerical computing',
        'librosa': 'Audio processing',
        'scipy': 'Advanced feature extraction',
        'sklearn': 'Machine learning',
        'fastapi': 'API framework',
        'pydub': 'Audio decoding'
    }
    
    all_ok = True
    for pkg, desc in required.items():
        try:
            __import__(pkg)
            print_status(f"{pkg:15s} - {desc}", 'success')
        except ImportError:
            print_status(f"{pkg:15s} - MISSING!", 'error')
            all_ok = False
    
    return all_ok


def check_file_structure():
    """Check if all required files exist"""
    print("\n" + "="*60)
    print("2. CHECKING FILE STRUCTURE")
    print("="*60)
    
    base_dir = Path(__file__).parent.parent
    
    required_files = {
        'Enhanced Scripts': [
            'scripts/collect_data_enhanced.py',
            'scripts/train_model_enhanced.py',
            'scripts/augment_data.py',
        ],
        'Advanced Features': [
            'app/services/feature_extractor_advanced.py',
        ],
        'Core Files': [
            'app/main.py',
            'app/services/classifier.py',
            'app/services/audio_decoder.py',
        ],
        'Data Directories': [
            'data/raw/ai',
            'data/raw/human',
        ],
        'Models': [
            'models/voice_classifier.pkl',
            'models/feature_scaler.pkl',
        ]
    }
    
    all_ok = True
    for category, files in required_files.items():
        print(f"\n{category}:")
        for file_path in files:
            full_path = base_dir / file_path
            if full_path.exists():
                if full_path.is_file():
                    size = full_path.stat().st_size
                    print_status(f"{file_path:50s} ({size:,} bytes)", 'success')
                else:
                    count = len(list(full_path.glob('*')))
                    print_status(f"{file_path:50s} ({count} files)", 'success')
            else:
                print_status(f"{file_path:50s} - MISSING", 'error')
                all_ok = False
    
    return all_ok


def test_advanced_features():
    """Test if advanced feature extraction works"""
    print("\n" + "="*60)
    print("3. TESTING ADVANCED FEATURE EXTRACTION")
    print("="*60)
    
    try:
        sys.path.append(str(Path(__file__).parent.parent))
        from app.services.feature_extractor_advanced import extract_advanced_features
        
        # Create dummy audio
        import librosa
        dummy_audio = np.random.randn(22050)  # 1 second
        
        print_status("Extracting features from dummy audio...", 'info')
        features = extract_advanced_features(dummy_audio, sr=22050)
        
        print_status(f"Extracted {len(features)} features", 'success')
        
        # Check for key features
        key_features = ['mfcc_0_mean', 'pitch_mean', 'jitter', 'shimmer', 
                       'hnr', 'mel_spec_mean', 'temporal_energy_variation']
        
        print("\nKey features present:")
        for feature in key_features:
            if feature in features:
                print_status(f"  {feature:30s} = {features[feature]:.4f}", 'success')
            else:
                print_status(f"  {feature:30s} - MISSING", 'error')
                return False
        
        return True
        
    except Exception as e:
        print_status(f"Feature extraction failed: {str(e)}", 'error')
        return False


def test_model_loading():
    """Test if trained model can be loaded"""
    print("\n" + "="*60)
    print("4. TESTING MODEL LOADING")
    print("="*60)
    
    try:
        import joblib
        base_dir = Path(__file__).parent.parent
        
        model_path = base_dir / 'models' / 'voice_classifier.pkl'
        scaler_path = base_dir / 'models' / 'feature_scaler.pkl'
        
        if not model_path.exists():
            print_status("Model file not found. Run: python scripts/train_model_enhanced.py", 'error')
            return False
        
        print_status("Loading model...", 'info')
        model = joblib.load(model_path)
        print_status(f"Model type: {type(model).__name__}", 'success')
        
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
            print_status(f"Scaler loaded: {type(scaler).__name__}", 'success')
        
        # Check model attributes
        if hasattr(model, 'n_estimators'):
            print_status(f"Number of estimators: {model.n_estimators}", 'info')
        if hasattr(model, 'feature_importances_'):
            print_status(f"Feature importances available: {len(model.feature_importances_)} features", 'info')
        
        return True
        
    except Exception as e:
        print_status(f"Model loading failed: {str(e)}", 'error')
        return False


def test_data_collection():
    """Check if data collection produced samples"""
    print("\n" + "="*60)
    print("5. CHECKING TRAINING DATA")
    print("="*60)
    
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data' / 'raw'
    
    ai_samples = list((data_dir / 'ai').glob('*.mp3')) if (data_dir / 'ai').exists() else []
    human_samples = list((data_dir / 'human').glob('*.mp3')) if (data_dir / 'human').exists() else []
    
    print_status(f"AI samples: {len(ai_samples)} files", 'success' if len(ai_samples) > 0 else 'warning')
    print_status(f"Human samples: {len(human_samples)} files", 'success' if len(human_samples) > 0 else 'warning')
    
    if len(ai_samples) == 0:
        print_status("Run: python scripts/collect_data_enhanced.py", 'warning')
    
    if len(human_samples) < 10:
        print_status("Add more human samples for better accuracy (target: 20+ per language)", 'warning')
    
    return len(ai_samples) > 0 and len(human_samples) > 0


def test_api_running():
    """Check if API is running"""
    print("\n" + "="*60)
    print("6. CHECKING API STATUS")
    print("="*60)
    
    try:
        import requests
        
        # Try local API
        print_status("Testing local API (localhost:8000)...", 'info')
        response = requests.get('http://localhost:8000/health', timeout=2)
        
        if response.status_code == 200:
            data = response.json()
            print_status(f"API Status: {data.get('status', 'unknown')}", 'success')
            print_status(f"Service: {data.get('service', 'unknown')}", 'success')
            return True
        else:
            print_status(f"API returned status {response.status_code}", 'warning')
            return False
            
    except requests.exceptions.ConnectionError:
        print_status("API not running locally", 'warning')
        print_status("Start with: python -m uvicorn app.main:app --reload --port 8000", 'info')
        return False
    except Exception as e:
        print_status(f"API check failed: {str(e)}", 'error')
        return False


def print_summary(results):
    """Print final summary"""
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    categories = [
        ('Dependencies', results['dependencies']),
        ('File Structure', results['files']),
        ('Advanced Features', results['features']),
        ('Model Loading', results['model']),
        ('Training Data', results['data']),
        ('API Status', results['api']),
    ]
    
    passed = sum(1 for _, result in categories if result)
    total = len(categories)
    
    print(f"\nTests Passed: {passed}/{total}\n")
    
    for name, result in categories:
        status = 'success' if result else 'error'
        print_status(f"{name:20s}: {'PASS' if result else 'FAIL'}", status)
    
    print("\n" + "="*60)
    
    if passed == total:
        print(f"{GREEN}✓ ALL SYSTEMS OPERATIONAL{RESET}")
        print("\nYour enhanced model is ready to use!")
        print("\nNext steps:")
        print("  1. Test API: python scripts/test_api.py")
        print("  2. Improve accuracy: Add real human voice samples")
        print("  3. Retrain: python scripts/train_model_enhanced.py")
    elif passed >= 4:
        print(f"{YELLOW}⚠ MOSTLY WORKING - Minor issues{RESET}")
        print("\nSome components need attention. See errors above.")
    else:
        print(f"{RED}✗ CRITICAL ISSUES DETECTED{RESET}")
        print("\nPlease fix the errors above before proceeding.")
    
    print("="*60)


def main():
    """Run all verification checks"""
    print("\n" + "="*60)
    print("AI VOICE DETECTION - SYSTEM VERIFICATION")
    print("="*60)
    
    results = {
        'dependencies': check_dependencies(),
        'files': check_file_structure(),
        'features': test_advanced_features(),
        'model': test_model_loading(),
        'data': test_data_collection(),
        'api': test_api_running(),
    }
    
    print_summary(results)


if __name__ == "__main__":
    main()
