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
    has_data = (data_dir / "ai").exists() and (data_dir / "human").exists()
    
    if not has_data:
        print("\n📢 No training data found - generating samples...")
        from scripts.collect_data import main as collect_data
        collect_data()
    
    # Train the model
    print("\n📢 Training model...")
    try:
        # Try enhanced training first
        from scripts.train_model_enhanced import main as train_enhanced
        train_enhanced()
    except Exception as e:
        print(f"⚠ Enhanced training failed: {e}")
        print("  Falling back to basic training...")
        try:
            from scripts.train_model import main as train_basic
            train_basic()
        except Exception as e2:
            print(f"✗ Basic training also failed: {e2}")
            print("  API will use heuristic classification")
            return
    
    print("\n✓ Model training complete!")
    print("="*60)


if __name__ == "__main__":
    main()
