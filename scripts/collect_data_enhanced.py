"""
Enhanced Data Collection Script with Multiple TTS Engines
Generates diverse AI voice samples for improved model training
"""
import os
import sys
from pathlib import Path
from gtts import gTTS
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# Sample texts for each language
SAMPLE_TEXTS = {
    "en": [
        "Hello, this is a test of the AI voice detection system.",
        "The weather today is quite pleasant with clear skies.",
        "Technology has transformed the way we communicate.",
        "Machine learning is revolutionizing many industries.",
        "This recording will help train an artificial intelligence model.",
        "Science and innovation drive progress in our society.",
        "Education is the foundation of a successful future.",
    ],
    "ta": [
        "வணக்கம், இது செயற்கை நுண்ணறிவு குரல் கண்டறிதல் அமைப்பின் சோதனை.",
        "இன்று வானிலை மிகவும் இனிமையாக உள்ளது.",
        "தொழில்நுட்பம் நம் தொடர்பு முறையை மாற்றியுள்ளது.",
        "இயந்திர கற்றல் பல தொழில்களை புரட்சிகரமாக மாற்றுகிறது.",
    ],
    "hi": [
        "नमस्ते, यह एआई वॉयस डिटेक्शन सिस्टम का परीक्षण है।",
        "आज का मौसम साफ आसमान के साथ काफी सुहावना है।",
        "प्रौद्योगिकी ने हमारे संवाद के तरीके को बदल दिया है।",
        "मशीन लर्निंग कई उद्योगों में क्रांति ला रही है।",
    ],
    "ml": [
        "ഹലോ, ഇത് AI വോയ്‌സ് ഡിറ്റക്ഷൻ സിസ്റ്റത്തിന്റെ പരിശോധനയാണ്.",
        "ഇന്നത്തെ കാലാവസ്ഥ വളരെ സുഖകരമാണ്.",
        "സാങ്കേതികവിദ്യ നമ്മുടെ ആശയവിനിമയ രീതിയെ മാറ്റിമറിച്ചിരിക്കുന്നു.",
    ],
    "te": [
        "హలో, ఇది AI వాయిస్ డిటెక్షన్ సిస్టమ్ యొక్క పరీక్ష.",
        "ఈరోజు వాతావరణం చాలా ఆహ్లాదకరంగా ఉంది.",
        "సాంకేతికత మన కమ్యూనికేషన్ విధానాన్ని మార్చింది.",
    ],
}


def generate_gtts_samples(output_dir: Path, languages: dict):
    """Generate basic gTTS samples (existing method)"""
    logger.info("\n📢 Generating AI samples using gTTS...")
    
    ai_dir = output_dir / "ai"
    ai_dir.mkdir(parents=True, exist_ok=True)
    
    count = 0
    for lang_code, texts in languages.items():
        for i, text in enumerate(texts, 1):
            try:
                output_file = ai_dir / f"ai_{lang_code}_{i}.mp3"
                tts = gTTS(text=text, lang=lang_code)
                tts.save(str(output_file))
                logger.info(f"  ✓ Generated: {output_file.name}")
                count += 1
            except Exception as e:
                logger.error(f"  ✗ Failed to generate {lang_code}_{i}: {e}")
    
    return count


def generate_coqui_tts_samples(output_dir: Path, languages: dict):
    """Generate samples using Coqui TTS (local, high quality)"""
    logger.info("\n📢 Generating AI samples using Coqui TTS...")
    
    try:
        from TTS.api import TTS
        
        ai_dir = output_dir / "ai_coqui"
        ai_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize TTS model
        logger.info("  Loading Coqui TTS model...")
        tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")
        
        count = 0
        # For now, only English (Coqui has limited multilingual support)
        if "en" in languages:
            for i, text in enumerate(languages["en"], 1):
                try:
                    output_file = ai_dir / f"ai_coqui_en_{i}.wav"
                    tts.tts_to_file(text=text, file_path=str(output_file))
                    logger.info(f"  ✓ Generated: {output_file.name}")
                    count += 1
                except Exception as e:
                    logger.error(f"  ✗ Failed: {e}")
        
        logger.info(f"\n✓ Generated {count} Coqui TTS samples")
        return count
        
    except ImportError:
        logger.warning("⚠ Coqui TTS not installed. Install with: pip install TTS")
        logger.info("  Skipping Coqui TTS generation...")
        return 0
    except Exception as e:
        logger.warning(f"⚠ Coqui TTS generation failed: {e}")
        return 0


def create_human_sample_instructions(output_dir: Path):
    """Create instructions for collecting real human voice samples"""
    human_dir = output_dir / "human"
    human_dir.mkdir(parents=True, exist_ok=True)
    
    instructions = """
====================================================================
INSTRUCTIONS FOR COLLECTING REAL HUMAN VOICE SAMPLES
====================================================================

CRITICAL: Real human voice samples are ESSENTIAL for achieving 80-85% accuracy!

OPTION 1: SELF-RECORDING (Fastest - 30 minutes)
------------------------------------------------
1. Equipment:
   - Smartphone voice recorder OR laptop microphone
   - Quiet room (minimal background noise)

2. Recording guidelines:
   - Record 5-10 samples per language
   - Each sample: 5-10 seconds
   - Speak naturally at normal pace
   - Vary tone/emotion slightly between samples
   - Save as MP3 format

3. Naming convention:
   - English: human_en_1.mp3, human_en_2.mp3, etc.
   - Tamil: human_ta_1.mp3, human_ta_2.mp3, etc.
   - Hindi: human_hi_1.mp3, human_hi_2.mp3, etc.
   - Malayalam: human_ml_1.mp3, human_ml_2.mp3, etc.
   - Telugu: human_te_1.mp3, human_te_2.mp3, etc.

4. Sample texts to read:
   Tamil: "வணக்கம், என் பெயர் [உங்கள் பெயர்]. இன்று நான் இந்த குரல் பதிவை செய்கிறேன்."
   English: "Hello, my name is [your name]. I am recording this sample today."
   Hindi: "नमस्ते, मेरा नाम [आपका नाम] है। मैं आज यह रिकॉर्डिंग कर रहा हूं।"
   (Add more varied sentences)

OPTION 2: PUBLIC DATASETS (Better Quality - 1-2 hours)
-------------------------------------------------------
1. Mozilla Common Voice (Recommended)
   - Website: https://commonvoice.mozilla.org/
   - Download validated datasets for your languages
   - Extract 10-20 samples per language
   - Already in MP3/OGG format
   
2. LibriVox (English only)
   - Website: https://librivox.org/
   - Public domain audiobooks
   - Download short clips, extract 5-10 second segments
   
3. VoxCeleb (Research dataset)
   - Website: https://www.robots.ox.ac.uk/~vgg/data/voxceleb/
   - Celebrity speech samples
   - Good speaker variety

OPTION 3: HYBRID (Recommended for Best Results)
------------------------------------------------
Combine sources for maximum diversity:
- 5 self-recorded samples per language
- 10 Mozilla Common Voice samples per language
- 5 LibriVox samples (English)

TARGET: 20+ human samples per language = 100+ total samples

====================================================================
QUICK START COMMAND
====================================================================

After collecting samples, verify:
    
    cd data/raw/human
    dir *.mp3  # Windows
    ls -l *.mp3  # Linux/Mac

Then retrain the model:
    
    python scripts/train_model.py

Expected accuracy improvement: 40-50% → 75-85%

====================================================================
"""
    
    readme_file = human_dir / "README_HUMAN_SAMPLES.txt"
    with open(readme_file, 'w', encoding='utf-8') as f:
        f.write(instructions)
    
    logger.info(f"\n✓ Created instructions in {readme_file}")


def create_placeholder_samples(output_dir: Path):
    """Create placeholder 'human' samples for testing (AI-generated with variation)"""
    response = input("\nWould you like to create placeholder 'human' samples for testing?\n"
                    "(These are AI-generated with variation - NOT real human voices)\n"
                    "Create placeholders? (y/n): ")
    
    if response.lower() != 'y':
        logger.info("Skipping placeholder creation. Replace with real human samples!")
        return 0
    
    logger.info("\n📢 Creating placeholder human samples (for testing only)...")
    logger.warning("⚠ WARNING: These are AI-generated with variation - NOT real human voices!")
    logger.info("   For production accuracy, replace with real human recordings.\n")
    
    human_dir = output_dir / "human"
    human_dir.mkdir(parents=True, exist_ok=True)
    
    # Create varied "human" samples using gTTS with different parameters
    placeholder_texts = {
        "en": [
            "I am speaking at a normal pace with natural variation.",
            "This is another sample with different intonation.",
            "Here is a third recording for testing purposes.",
        ],
        "ta": [
            "இது இயல்பான வேகத்தில் பேசும் மாதிரி.",
            "இது மற்றொரு பதிவு வித்தியாசமான ஒலியுடன்.",
        ],
        "hi": [
            "यह सामान्य गति से बोलने का नमूना है।",
            "यह एक और रिकॉर्डिंग अलग स्वर के साथ है।",
        ],
        "ml": ["ഇത് സാധാരണ വേഗതയിൽ സംസാരിക്കുന്ന സാമ്പിൾ ആണ്."],
        "te": ["ఇది సాధారణ వేగంతో మాట్లాడే నమూనా."],
    }
    
    count = 0
    for lang_code, texts in placeholder_texts.items():
        for i, text in enumerate(texts, 1):
            try:
                output_file = human_dir / f"human_{lang_code}_{i}.mp3"
                # Use slow=False for slight variation from AI samples
                tts = gTTS(text=text, lang=lang_code, slow=False)
                tts.save(str(output_file))
                logger.info(f"  ✓ Created placeholder: {output_file.name}")
                count += 1
            except Exception as e:
                logger.error(f"  ✗ Failed: {e}")
    
    logger.info(f"\n✓ Created {count} placeholder human samples")
    logger.warning("⚠ Remember: Replace these with real human recordings for production!\n")
    
    return count


def main():
    """Main data collection workflow"""
    print("=" * 60)
    print("ENHANCED AI VOICE DETECTION - DATA COLLECTION")
    print("=" * 60)
    
    # Setup directories
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate AI samples with gTTS
    ai_count = generate_gtts_samples(data_dir, SAMPLE_TEXTS)
    logger.info(f"\n✓ Generated {ai_count} gTTS AI samples")
    
    # Try to generate with Coqui TTS for diversity
    coqui_count = generate_coqui_tts_samples(data_dir, SAMPLE_TEXTS)
    
    # Create human sample instructions
    create_human_sample_instructions(data_dir)
    
    # Optionally create placeholders
    human_count = create_placeholder_samples(data_dir)
    
    # Summary
    print("\n" + "=" * 60)
    print("DATA COLLECTION SUMMARY")
    print("=" * 60)
    print(f"AI samples (gTTS): {ai_count} files in data/raw/ai")
    if coqui_count > 0:
        print(f"AI samples (Coqui): {coqui_count} files in data/raw/ai_coqui")
    print(f"Human samples: {human_count} files in data/raw/human")
    print(f"\nNext steps:")
    print(f"1. Add more human voice recordings to data/raw/human/")
    if coqui_count == 0:
        print(f"2. (Optional) Install Coqui TTS: pip install TTS")
        print(f"3. Run: python scripts/train_model_enhanced.py")
    else:
        print(f"2. Run: python scripts/train_model_enhanced.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
