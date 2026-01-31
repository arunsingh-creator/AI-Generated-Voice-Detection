"""
Data collection script - Downloads human voice samples and generates AI samples

This script:
1. Downloads public domain human speech samples
2. Generates AI voice samples using gTTS (Google Text-to-Speech)
3. Organizes data for training
"""
import os
import requests
from gtts import gTTS
import random
from pathlib import Path

# Create data directories
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
HUMAN_DIR = RAW_DIR / "human"
AI_DIR = RAW_DIR / "ai"

for dir_path in [DATA_DIR, RAW_DIR, HUMAN_DIR, AI_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)


# Sample texts for different languages
SAMPLE_TEXTS = {
    "en": [
        "Hello, this is a sample of human speech for testing purposes.",
        "The quick brown fox jumps over the lazy dog near the riverbank.",
        "Artificial intelligence is transforming the way we live and work.",
        "Machine learning models can detect patterns in audio signals.",
        "Natural language processing helps computers understand human speech.",
    ],
    "ta": [
        "வணக்கம், இது சோதனை நோக்கங்களுக்கான மனித பேச்சின் மாதிரி.",
        "செயற்கை நுண்ணறிவு நம் வாழ்க்கை மற்றும் வேலை முறையை மாற்றுகிறது.",
    ],
    "hi": [
        "नमस्ते, यह परीक्षण उद्देश्यों के लिए मानव भाषण का एक नमूना है.",
        "कृत्रिम बुद्धिमत्ता हमारे जीवन और काम करने के तरीके को बदल रही है.",
    ],
    "ml": [
        "ഹലോ, ഇത് പരിശോധനാ ആവശ്യങ്ങൾക്കുള്ള മനുഷ്യ സംഭാഷണത്തിന്റെ മാതൃകയാണ്.",
    ],
    "te": [
        "హలో, ఇది పరీక్ష ప్రయోజనాల కోసం మానవ ప్రసంగం యొక్క నమూనా.",
    ],
}


def generate_ai_samples():
    """Generate AI voice samples using gTTS"""
    print("📢 Generating AI voice samples using gTTS...")
    
    sample_count = 0
    for lang_code, texts in SAMPLE_TEXTS.items():
        for i, text in enumerate(texts):
            try:
                output_file = AI_DIR / f"ai_{lang_code}_{i+1}.mp3"
                
                if output_file.exists():
                    print(f"  ⏭ Skipping {output_file.name} (already exists)")
                    continue
                
                # Generate TTS audio
                tts = gTTS(text=text, lang=lang_code, slow=False)
                tts.save(str(output_file))
                
                sample_count += 1
                print(f"  ✓ Generated: {output_file.name}")
                
            except Exception as e:
                print(f"  ✗ Failed to generate {lang_code}_{i+1}: {e}")
    
    print(f"\n✓ Generated {sample_count} AI voice samples\n")


def download_human_samples():
    """
    Download human voice samples from public sources
    
    Note: For production, use proper datasets like:
    - LibriVox (public domain audiobooks)
    - Common Voice (Mozilla)
    - VoxForge
    
    This is a placeholder that generates instructions for manual collection.
    """
    print("📢 Human voice sample collection instructions:\n")
    print("For best results, collect human voice samples manually:")
    print("1. Record your own voice samples (5-10 seconds each)")
    print("2. Use public domain sources like LibriVox")
    print("3. Use Mozilla Common Voice dataset")
    print("4. Ensure you have samples in all 5 languages: Tamil, English, Hindi, Malayalam, Telugu")
    print(f"\nPlace MP3 files in: {HUMAN_DIR.absolute()}\n")
    print("File naming: human_en_1.mp3, human_ta_1.mp3, etc.\n")
    
    # Create placeholder files with instructions
    readme_path = HUMAN_DIR / "README.txt"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("HUMAN VOICE SAMPLES\n")
        f.write("=" * 50 + "\n\n")
        f.write("Place your human voice MP3 samples here.\n\n")
        f.write("Naming convention:\n")
        f.write("  - human_en_1.mp3, human_en_2.mp3 (English)\n")
        f.write("  - human_ta_1.mp3, human_ta_2.mp3 (Tamil)\n")
        f.write("  - human_hi_1.mp3, human_hi_2.mp3 (Hindi)\n")
        f.write("  - human_ml_1.mp3 (Malayalam)\n")
        f.write("  - human_te_1.mp3 (Telugu)\n\n")
        f.write("Recommended sources:\n")
        f.write("  - Record your own voice (5-10 second clips)\n")
        f.write("  - LibriVox: https://librivox.org/\n")
        f.write("  - Mozilla Common Voice: https://commonvoice.mozilla.org/\n")
        f.write("  - VoxForge: http://www.voxforge.org/\n\n")
        f.write("Aim for at least 10-20 samples per language for decent training.\n")
    
    print(f"✓ Created instructions in {readme_path}\n")


def create_simple_human_samples():
    """
    Create a few basic human-like samples for initial testing
    Using gTTS with speed variation to simulate human characteristics
    
    Note: These are still AI-generated but with variations.
    For production, replace with actual human recordings.
    """
    print("📢 Creating placeholder human samples (for testing only)...")
    print("⚠ WARNING: These are AI-generated with variation - NOT real human voices!")
    print("   For production accuracy, replace with real human recordings.\n")
    
    sample_count = 0
    for lang_code, texts in SAMPLE_TEXTS.items():
        for i in range(min(3, len(texts))):  # Create 3 samples per language
            try:
                output_file = HUMAN_DIR / f"human_{lang_code}_{i+1}.mp3"
                
                if output_file.exists():
                    print(f"  ⏭ Skipping {output_file.name} (already exists)")
                    continue
                
                # Use slow=True for variation
                tts = gTTS(text=texts[i], lang=lang_code, slow=(i % 2 == 0))
                tts.save(str(output_file))
                
                sample_count += 1
                print(f"  ✓ Created placeholder: {output_file.name}")
                
            except Exception as e:
                print(f"  ✗ Failed to create {lang_code}_{i+1}: {e}")
    
    print(f"\n✓ Created {sample_count} placeholder human samples")
    print("⚠ Remember: Replace these with real human recordings for production!\n")


def main():
    """Main data collection workflow"""
    print("=" * 60)
    print("AI VOICE DETECTION - DATA COLLECTION")
    print("=" * 60)
    print()
    
    # Generate AI samples using TTS
    generate_ai_samples()
    
    # Provide instructions for human samples
    download_human_samples()
    
    # Create placeholder human samples for testing
    print("Would you like to create placeholder 'human' samples for testing?")
    print("(These are AI-generated with variation - NOT real human voices)")
    response = input("Create placeholders? (y/n): ").strip().lower()
    
    if response == 'y':
        create_simple_human_samples()
    
    print("\n" + "=" * 60)
    print("DATA COLLECTION SUMMARY")
    print("=" * 60)
    
    ai_files = list(AI_DIR.glob("*.mp3"))
    human_files = list(HUMAN_DIR.glob("*.mp3"))
    
    print(f"AI samples: {len(ai_files)} files in {AI_DIR}")
    print(f"Human samples: {len(human_files)} files in {HUMAN_DIR}")
    
    if len(human_files) < 5:
        print("\n⚠ WARNING: Very few human samples!")
        print("  For good accuracy, collect at least 10-20 human voice samples.")
    
    print("\nNext steps:")
    print("1. Add more human voice recordings to data/raw/human/")
    print("2. Run: python scripts/train_model.py")
    print()


if __name__ == "__main__":
    main()
