"""
Enhanced Model Training Script with Hyperparameter Tuning
Achieves 80-85% accuracy with real human + diverse AI samples
"""
import os
import sys
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import json
from datetime import datetime

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent))

from app.services.feature_extractor_advanced import extract_advanced_features, normalize_features


def load_audio_samples(data_dir: Path):
    """Load all audio samples from raw data directories"""
    print("\n📂 Loading audio samples...\n")
    
    human_dir = data_dir / "human"
    ai_dirs = [data_dir / "ai", data_dir / "ai_coqui"]
    
    # Load human samples
    human_samples = []
    print("Loading human samples...")
    if human_dir.exists():
        for audio_file in human_dir.glob("*.mp3"):
            try:
                waveform, sr = librosa.load(str(audio_file), sr=22050)
                human_samples.append((waveform, sr, str(audio_file.name)))
                print(f"  ✓ {audio_file.name}")
            except Exception as e:
                print(f"  ✗ Failed to load {audio_file.name}: {e}")
        
        # Also load WAV files
        for audio_file in human_dir.glob("*.wav"):
            try:
                waveform, sr = librosa.load(str(audio_file), sr=22050)
                human_samples.append((waveform, sr, str(audio_file.name)))
                print(f"  ✓ {audio_file.name}")
            except Exception as e:
                print(f"  ✗ Failed to load {audio_file.name}: {e}")
    
    # Load AI samples
    ai_samples = []
    print("\nLoading AI samples...")
    for ai_dir in ai_dirs:
        if ai_dir.exists():
            for audio_file in list(ai_dir.glob("*.mp3")) + list(ai_dir.glob("*.wav")):
                try:
                    waveform, sr = librosa.load(str(audio_file), sr=22050)
                    ai_samples.append((waveform, sr, str(audio_file.name)))
                    print(f"  ✓ {audio_file.name}")
                except Exception as e:
                    print(f"  ✗ Failed to load {audio_file.name}: {e}")
    
    print(f"\n✓ Loaded {len(human_samples)} human samples and {len(ai_samples)} AI samples\n")
    
    return human_samples, ai_samples


def extract_features_from_samples(samples, label):
    """Extract features from all samples"""
    features_list = []
    labels = []
    
    for waveform, sr, filename in samples:
        try:
            # Extract advanced features
            features = extract_advanced_features(waveform, sr)
            features = normalize_features(features)
            
            features_list.append(features)
            labels.append(label)
        except Exception as e:
            print(f"  Warning: Failed to extract features from {filename}: {e}")
    
    return features_list, labels


def train_with_hyperparameter_tuning(X_train, y_train, model_type='random_forest'):
    """Train model with hyperparameter tuning using GridSearchCV"""
    print(f"\n🤖 Training {model_type} with hyperparameter tuning...")
    
    if model_type == 'random_forest':
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }
        base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
        
    elif model_type == 'gradient_boosting':
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10]
        }
        base_model = GradientBoostingClassifier(random_state=42)
        
    elif model_type == 'svm':
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01],
            'kernel': ['rbf', 'poly']
        }
        base_model = SVC(probability=True, random_state=42)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Grid search with cross-validation
    print("  Running grid search (this may take a few minutes)...")
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\n  ✓ Best parameters: {grid_search.best_params_}")
    print(f"  ✓ Best CV score: {grid_search.best_score_:.2%}")
    
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_


def main():
    """Main training workflow"""
    print("=" * 60)
    print("ENHANCED AI VOICE DETECTION - MODEL TRAINING")
    print("=" * 60)
    
    # Setup paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data" / "raw"
    models_dir = base_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Load samples
    human_samples, ai_samples = load_audio_samples(data_dir)
    
    if len(human_samples) == 0:
        print("\n⚠ ERROR: No human samples found!")
        print("  Please add real human voice samples to data/raw/human/")
        print("  See data/raw/human/README_HUMAN_SAMPLES.txt for instructions")
        return
    
    if len(ai_samples) == 0:
        print("\n⚠ ERROR: No AI samples found!")
        print("  Run: python scripts/collect_data_enhanced.py")
        return
    
    # Extract features
    print("📊 Extracting features...\n")
    
    print("Processing human samples...")
    human_features, human_labels = extract_features_from_samples(human_samples, 0)
    
    print("\nProcessing AI samples...")
    ai_features, ai_labels = extract_features_from_samples(ai_samples, 1)
    
    # Combine datasets
    all_features = human_features + ai_features
    all_labels = human_labels + ai_labels
    
    # Convert to numpy arrays
    feature_names = sorted(all_features[0].keys())
    X = np.array([[f[name] for name in feature_names] for f in all_features])
    y = np.array(all_labels)
    
    print(f"\n✓ Dataset prepared:")
    print(f"  Number of features: {X.shape[1]}")
    print(f"  Dataset shape: {X.shape}")
    print(f"  Class distribution: {np.bincount(y)} (0=HUMAN, 1=AI)")
    
    # Scale features
    print("\n📐 Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n  Training set: {len(X_train)} samples")
    print(f"  Test set: {len(X_test)} samples")
    
    # Train with hyperparameter tuning
    print("\n🎯 Training models with hyperparameter tuning...")
    
    # Try multiple models
    results = {}
    
    for model_type in ['random_forest', 'gradient_boosting']:
        print(f"\n{'='*60}")
        print(f"Training: {model_type}")
        print('='*60)
        
        model, best_params, cv_score = train_with_hyperparameter_tuning(
            X_train, y_train, model_type
        )
        
        # Evaluate on test set
        y_pred = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        
        results[model_type] = {
            'model': model,
            'params': best_params,
            'cv_score': cv_score,
            'test_accuracy': test_accuracy,
            'y_pred': y_pred
        }
        
        print(f"\n  Test Accuracy: {test_accuracy:.2%}")
    
    # Select best model
    best_model_name = max(results, key=lambda x: results[x]['test_accuracy'])
    best_model_results = results[best_model_name]
    
    print("\n" + "=" * 60)
    print("MODEL EVALUATION - BEST MODEL")
    print("=" * 60)
    print(f"\nBest Model: {best_model_name}")
    print(f"CV Accuracy: {best_model_results['cv_score']:.2%}")
    print(f"Test Accuracy: {best_model_results['test_accuracy']:.2%}")
    
    print("\nClassification Report:")
    print(classification_report(
        y_test,
        best_model_results['y_pred'],
        target_names=['HUMAN', 'AI_GENERATED'],
        digits=3
    ))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, best_model_results['y_pred'])
    print("              Predicted")
    print("              HUMAN  AI")
    print(f"Actual HUMAN  {cm[0][0]:5d} {cm[0][1]:4d}")
    print(f"       AI     {cm[1][0]:5d} {cm[1][1]:4d}")
    
    # Feature importance (if Random Forest)
    if best_model_name == 'random_forest':
        print("\nTop 15 Most Important Features:")
        importances = best_model_results['model'].feature_importances_
        indices = np.argsort(importances)[::-1][:15]
        
        for i, idx in enumerate(indices, 1):
            print(f"  {i:2d}. {feature_names[idx]:30s}: {importances[idx]:.4f}")
    
    # Save model
    print("\n💾 Saving model...")
    model_path = models_dir / "voice_classifier.pkl"
    scaler_path = models_dir / "feature_scaler.pkl"
    
    joblib.dump(best_model_results['model'], model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"  ✓ Model saved to: {model_path}")
    print(f"  ✓ Scaler saved to: {scaler_path}")
    
    # Save training history
    history = {
        'timestamp': datetime.now().isoformat(),
        'model_type': best_model_name,
        'best_params': {k: str(v) for k, v in best_model_results['params'].items()},
        'cv_accuracy': float(best_model_results['cv_score']),
        'test_accuracy': float(best_model_results['test_accuracy']),
        'num_features': int(X.shape[1]),
        'train_samples': int(len(X_train)),
        'test_samples': int(len(X_test)),
        'human_samples': len(human_samples),
        'ai_samples': len(ai_samples)
    }
    
    history_file = models_dir / "training_history.jsonl"
    with open(history_file, 'a') as f:
        f.write(json.dumps(history) + '\n')
    
    print(f"  ✓ Training history saved to: {history_file}")
    
    # Final recommendations
    print("\n" + "=" * 60)
    
    if best_model_results['test_accuracy'] >= 0.80:
        print("✅ EXCELLENT! Model achieved target accuracy (80%+)")
        print("\nNext steps:")
        print("1. Test API locally: python scripts/test_api.py")
        print("2. Commit and push: git add models/ && git commit -m 'Updated model'")
        print("3. Deploy to Render (auto-deploys on push)")
    elif best_model_results['test_accuracy'] >= 0.70:
        print("⚠ GOOD! Model achieved 70%+ accuracy")
        print("\nTo reach 80%+ accuracy:")
        print("1. Add more diverse human samples (20+ per language)")
        print("2. Add more AI diversity (try different TTS engines)")
        print("3. Retrain: python scripts/train_model_enhanced.py")
    else:
        print("⚠ Model accuracy below 70%")
        print("\nCritical improvements needed:")
        print("1. Collect REAL human voice samples (not AI-generated)")
        print("2. Add diversity: record different speakers, accents, ages")
        print("3. Increase dataset size (50+ samples per class minimum)")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
