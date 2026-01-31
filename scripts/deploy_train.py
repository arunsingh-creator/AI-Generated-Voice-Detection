"""
Deployment Model Training Script
Automatically trains the model when deployed to production
"""
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def main():
    print("\n" + "="*60)
    print("DEPLOYMENT MODEL TRAINING")
    print("="*60)
    
    # Check if models already exist
    models_dir = Path("models")
    model_path = models_dir / "voice_classifier.pkl"
    scaler_path = models_dir / "feature_scaler.pkl"
    
    if model_path.exists() and scaler_path.exists():
        print("✓ Models already exist, skipping training")
        print(f"  Model: {model_path}")
        print(f"  Scaler: {scaler_path}")
        return
    
    print("\n⚠ Models not found - running training...")
    
    
    # Check if we have training data
    data_dir = Path("data/raw")
    ai_dir = data_dir / "ai"
    human_dir = data_dir / "human"
    
    # Count actual audio files
    ai_count = len(list(ai_dir.glob("*.mp3"))) if ai_dir.exists() else 0
    human_count = len(list(human_dir.glob("*.mp3"))) if human_dir.exists() else 0
    
    if ai_count == 0 or human_count == 0:
        print(f"\n📢 Training data insufficient (AI: {ai_count}, Human: {human_count})")
        print("   Generating samples with gTTS...")
        try:
            from scripts.collect_data import main as collect_data
            collect_data()
        except Exception as e:
            print(f"⚠ Sample generation failed: {e}")
    
    # Train the model using basic training (faster for deployment)
    print("\n📢 Training model with basic trainer...")
    try:
        from scripts.train_model import main as train_basic
        train_basic()
        print("✓ Model trained successfully!")
    except Exception as e:
        print(f"⚠ Training failed: {e}")
        print("  API will use heuristic classification")
    
    print("\n✓ Model training complete!")
    print("="*60)


if __name__ == "__main__":
    main()
