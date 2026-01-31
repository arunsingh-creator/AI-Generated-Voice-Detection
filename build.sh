#!/bin/sh
# Render build script - runs during deployment

echo "🚀 Starting optimized build..."

# Pre-generate model during build phase (not runtime)
# This makes cold starts faster
echo "📦 Pre-generating model..."
python scripts/deploy_train.py

echo "✓ Build complete - service will start fast!"
