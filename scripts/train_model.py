"""
Model training script - Trains classifier on collected voice samples

This script:
1. Loads audio samples from data/raw/
2. Extracts features using the feature extractor
3. Trains a Random Forest classifier
4. Evaluates model performance
5. Saves the trained model
"""
import os
import sys
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import librosa

# Add parent directory to path to import app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.feature_extractor import extract_features
from app.config import settings


def load_audio_samples():
    """Load and extract features from all audio samples"""
    print("📂 Loading audio samples...\n")
    
    data_dir = Path("data/raw")
    human_dir = data_dir / "human"
    ai_dir = data_dir / "ai"
    
    features_list = []
    labels_list = []
    
    # Load human samples (label = 0)
    print("Loading human samples...")
    human_files = list(human_dir.glob("*.mp3"))
    for audio_file in human_files:
        try:
            # Load audio
            waveform, sr = librosa.load(str(audio_file), sr=settings.SAMPLE_RATE)
            
            # Extract features
            features = extract_features(waveform, sr)
            features_list.append(features)
            labels_list.append(0)  # 0 = HUMAN
            
            print(f"  ✓ {audio_file.name}")
        except Exception as e:
            print(f"  ✗ Failed to load {audio_file.name}: {e}")
    
    # Load AI samples (label = 1)
    print("\nLoading AI samples...")
    ai_files = list(ai_dir.glob("*.mp3"))
    for audio_file in ai_files:
        try:
            # Load audio
            waveform, sr = librosa.load(str(audio_file), sr=settings.SAMPLE_RATE)
            
            # Extract features
            features = extract_features(waveform, sr)
            features_list.append(features)
            labels_list.append(1)  # 1 = AI_GENERATED
            
            print(f"  ✓ {audio_file.name}")
        except Exception as e:
            print(f"  ✗ Failed to load {audio_file.name}: {e}")
    
    print(f"\n✓ Loaded {len(human_files)} human samples and {len(ai_files)} AI samples")
    
    return features_list, labels_list


def prepare_dataset(features_list, labels_list):
    """Convert feature dictionaries to numpy arrays"""
    print("\n📊 Preparing dataset...")
    
    # Get feature names from first sample
    feature_names = sorted(features_list[0].keys())
    print(f"  Number of features: {len(feature_names)}")
    
    # Convert to numpy array
    X = np.array([
        [features[name] for name in feature_names]
        for features in features_list
    ])
    y = np.array(labels_list)
    
    print(f"  Dataset shape: {X.shape}")
    print(f"  Class distribution: {np.bincount(y)} (0=HUMAN, 1=AI)")
    
    return X, y, feature_names


def train_model(X, y):
    """Train Random Forest classifier"""
    print("\n🤖 Training model...\n")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples\n")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest
    print("Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, y_train)
    print("✓ Training complete\n")
    
    # Evaluate
    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    print(f"\n5-Fold Cross-Validation Accuracy: {cv_scores.mean():.2%} (+/- {cv_scores.std():.2%})")
    
    # Test set evaluation
    y_pred = model.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nTest Set Accuracy: {test_accuracy:.2%}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=['HUMAN', 'AI_GENERATED']
    ))
    
    # Confusion matrix
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"              Predicted")
    print(f"              HUMAN  AI")
    print(f"Actual HUMAN    {cm[0][0]:3d}   {cm[0][1]:3d}")
    print(f"       AI       {cm[1][0]:3d}   {cm[1][1]:3d}")
    
    print("\n" + "=" * 60)
    
    # Feature importance
    feature_importance = model.feature_importances_
    print("\nTop 10 Most Important Features:")
    # We'll just print top features (feature names would require passing them)
    top_indices = np.argsort(feature_importance)[-10:][::-1]
    for i, idx in enumerate(top_indices, 1):
        print(f"  {i}. Feature {idx}: {feature_importance[idx]:.4f}")
    
    return model, scaler, test_accuracy


def save_model(model, scaler):
    """Save trained model and scaler"""
    print("\n💾 Saving model...")
    
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Save model
    model_path = models_dir / "voice_classifier.pkl"
    joblib.dump(model, model_path)
    print(f"  ✓ Model saved to: {model_path}")
    
    # Save scaler
    scaler_path = models_dir / "feature_scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"  ✓ Scaler saved to: {scaler_path}")


def main():
    """Main training workflow"""
    print("=" * 60)
    print("AI VOICE DETECTION - MODEL TRAINING")
    print("=" * 60)
    print()
    
    # Check if data exists
    if not Path("data/raw/human").exists() or not Path("data/raw/ai").exists():
        print("❌ Error: Data directories not found!")
        print("   Run: python scripts/collect_data.py first")
        return
    
    # Load data
    features_list, labels_list = load_audio_samples()
    
    if len(features_list) < 10:
        print("\n⚠ WARNING: Very few samples detected!")
        print(f"   Found only {len(features_list)} samples.")
        print("   For good accuracy, collect at least 20-30 samples per class.")
        response = input("\nContinue anyway? (y/n): ").strip().lower()
        if response != 'y':
            print("Training cancelled.")
            return
    
    # Prepare dataset
    X, y, feature_names = prepare_dataset(features_list, labels_list)
    
    # Train model
    model, scaler, accuracy = train_model(X, y)
    
    # Provide feedback on accuracy
    print("\n" + "=" * 60)
    if accuracy >= 0.85:
        print("✓ EXCELLENT: Model accuracy is above 85%!")
        print("  This should perform well in production.")
    elif accuracy >= 0.75:
        print("✓ GOOD: Model accuracy is acceptable (75-85%)")
        print("  Consider collecting more diverse samples to improve.")
    else:
        print("⚠ WARNING: Model accuracy is below 75%")
        print("  Strongly recommend collecting more training data.")
        print("  Mix of real human voices and AI-generated samples needed.")
    
    # Save model
    save_model(model, scaler)
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("Next steps:")
    print("1. Copy .env.example to .env")
    print("2. Set your API key in .env")
    print("3. Run: python -m uvicorn app.main:app --reload")
    print("=" * 60)


if __name__ == "__main__":
    main()
