#!/usr/bin/env bash
# exit on error
set -o errexit

echo "🚀 Starting Native Build..."

# 1. Install Python Dependencies
echo "📦 Installing dependencies from requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt

# 2. Check/Setup FFMPEG (Native environment might need this)
if ! command -v ffmpeg &> /dev/null; then
    echo "⚠ FFMPEG not found! Attempting to install static build..."
    # Create bin directory
    mkdir -p bin
    # Download static build
    curl -L https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz -o ffmpeg.tar.xz
    tar -xf ffmpeg.tar.xz
    # Move binary to local bin
    mv ffmpeg-master-latest-linux64-gpl/bin/ffmpeg bin/
    mv ffmpeg-master-latest-linux64-gpl/bin/ffprobe bin/
    # Cleanup
    rm -rf ffmpeg-master-latest-linux64-gpl ffmpeg.tar.xz
    # Add to PATH
    export PATH="$PWD/bin:$PATH"
    echo "✓ FFMPEG installed locally"
else
    echo "✓ FFMPEG found at $(which ffmpeg)"
fi

# 3. Create necessary directories
echo "📂 Creating directory structure..."
mkdir -p models data/raw/ai data/raw/human

# 4. Run Auto-Training (Deployment Training)
echo "🧠 Running deployment training..."
python scripts/deploy_train.py

echo "✓ Build complete!"
