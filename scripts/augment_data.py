"""
Data Augmentation Pipeline for Voice Samples
Increases training data diversity through audio transformations
"""
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def augment_audio(audio_path: Path, output_dir: Path, num_augmentations=3):
    """
    Create augmented versions of an audio sample
    
    Augmentations:
    1. Time stretching (speed variation)
    2. Pitch shifting
    3. Adding slight noise
    4. Volume adjustment
    
    Args:
        audio_path: Path to original audio file
        output_dir: Directory to save augmented samples
        num_augmentations: Number of variations to create
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load original audio
    try:
        y, sr = librosa.load(str(audio_path), sr=22050)
    except Exception as e:
        logger.error(f"Failed to load {audio_path}: {e}")
        return 0
    
    base_name = audio_path.stem
    file_ext = audio_path.suffix
    created_count = 0
    
    # 1. Time stretching (slightly faster/slower)
    try:
        if num_augmentations >= 1:
            # Slightly slower (0.9x speed)
            y_slow = librosa.effects.time_stretch(y=y, rate=0.9)
            output_file = output_dir / f"{base_name}_slow{file_ext}"
            sf.write(str(output_file), y_slow, sr)
            logger.info(f"    ✓ Created: {output_file.name}")
            created_count += 1
    except Exception as e:
        logger.warning(f"    ⚠ Time stretching failed: {e}")
    
    # 2. Pitch shifting
    try:
        if num_augmentations >= 2:
            # Shift pitch up by 2 semitones
            y_pitched = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=2)
            output_file = output_dir / f"{base_name}_pitched{file_ext}"
            sf.write(str(output_file), y_pitched, sr)
            logger.info(f"    ✓ Created: {output_file.name}")
            created_count += 1
    except Exception as e:
        logger.warning(f"    ⚠ Pitch shifting failed: {e}")
    
    # 3. Add slight background noise
    try:
        if num_augmentations >= 3:
            # Add very subtle noise
            noise = np.random.normal(0, 0.003, len(y))
            y_noisy = y + noise
            # Ensure audio stays in valid range
            y_noisy = np.clip(y_noisy, -1.0, 1.0)
            output_file = output_dir / f"{base_name}_noisy{file_ext}"
            sf.write(str(output_file), y_noisy, sr)
            logger.info(f"    ✓ Created: {output_file.name}")
            created_count += 1
    except Exception as e:
        logger.warning(f"    ⚠ Noise addition failed: {e}")
    
    # 4. Volume adjustment
    try:
        if num_augmentations >= 4:
            # Increase volume by 20%
            y_louder = y * 1.2
            # Ensure audio stays in valid range
            y_louder = np.clip(y_louder, -1.0, 1.0)
            output_file = output_dir / f"{base_name}_louder{file_ext}"
            sf.write(str(output_file), y_louder, sr)
            logger.info(f"    ✓ Created: {output_file.name}")
            created_count += 1
    except Exception as e:
        logger.warning(f"    ⚠ Volume adjustment failed: {e}")
    
    return created_count


def augment_dataset(data_dir: Path, augment_human=True, augment_ai=False):
    """
    Augment entire dataset
    
    Args:
        data_dir: Path to data/raw directory
        augment_human: Whether to augment human samples (recommended)
        augment_ai: Whether to augment AI samples (usually not needed)
    """
    logger.info("=" * 60)
    logger.info("DATA AUGMENTATION PIPELINE")
    logger.info("=" * 60)
    
    total_created = 0
    
    # Augment human samples (important for balancing)
    if augment_human:
        logger.info("\n📢 Augmenting HUMAN samples...")
        human_dir = data_dir / "human"
        augmented_dir = data_dir / "human_augmented"
        
        if human_dir.exists():
            audio_files = list(human_dir.glob("*.mp3")) + list(human_dir.glob("*.wav"))
            logger.info(f"  Found {len(audio_files)} human samples\n")
            
            for audio_file in audio_files:
                logger.info(f"  Augmenting: {audio_file.name}")
                count = augment_audio(audio_file, augmented_dir, num_augmentations=3)
                total_created += count
        else:
            logger.warning("  ⚠ Human samples directory not found")
    
    # Augment AI samples (optional)
    if augment_ai:
        logger.info("\n📢 Augmenting AI samples...")
        ai_dir = data_dir / "ai"
        augmented_dir = data_dir / "ai_augmented"
        
        if ai_dir.exists():
            audio_files = list(ai_dir.glob("*.mp3")) + list(ai_dir.glob("*.wav"))
            logger.info(f"  Found {len(audio_files)} AI samples\n")
            
            for audio_file in audio_files:
                logger.info(f"  Augmenting: {audio_file.name}")
                count = augment_audio(audio_file, augmented_dir, num_augmentations=2)
                total_created += count
        else:
            logger.warning("  ⚠ AI samples directory not found")
    
    logger.info("\n" + "=" * 60)
    logger.info(f"✓ Created {total_created} augmented samples")
    logger.info("=" * 60)
    logger.info("\nNext steps:")
    logger.info("1. Move augmented samples to main directories:")
    logger.info("   - data/raw/human_augmented/*.mp3 → data/raw/human/")
    logger.info("   - data/raw/ai_augmented/*.mp3 → data/raw/ai/")
    logger.info("2. Retrain model: python scripts/train_model_enhanced.py")
    logger.info("=" * 60)
    
    return total_created


def main():
    """Main augmentation workflow"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Augment voice dataset")
    parser.add_argument(
        '--human-only',
        action='store_true',
        help='Only augment human samples (recommended)'
    )
    parser.add_argument(
        '--ai-only',
        action='store_true',
        help='Only augment AI samples'
    )
    parser.add_argument(
        '--both',
        action='store_true',
        help='Augment both human and AI samples'
    )
    
    args = parser.parse_args()
    
    # Determine what to augment
    if args.both:
        augment_human = True
        augment_ai = True
    elif args.ai_only:
        augment_human = False
        augment_ai = True
    else:  # Default or human-only
        augment_human = True
        augment_ai = False
    
    # Setup paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data" / "raw"
    
    # Run augmentation
    augment_dataset(data_dir, augment_human, augment_ai)


if __name__ == "__main__":
    main()
